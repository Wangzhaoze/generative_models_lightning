# 统一生成框架重组方案

## Summary
- 现状评价：你现在按 `diffusion/flow/gan/vae` 分目录是对的，`diffusion process` 和 `LightningModule` 分开也是对的，说明你已经有“算法层”和“训练层”分离的意识。
- 主要风险 1：根包同时承载公共导出和基类，实现放在 [__init__.py](d:/Projects/generative_models_lightining/generative_models_lightning/__init__.py#L13) 会让后续协议膨胀、导入边界变糊。
- 主要风险 2：条件生成还没有正式协议，但扩散代码已经隐含出 `model_kwargs`、`y`、`cond_fn`、`s` 这套松散接口；其中无条件路径在 [gaussian_diffusion.py](d:/Projects/generative_models_lightining/generative_models_lightning/diffusion/process/gaussian_diffusion.py#L926) 里实际默认要求 `y`，后面很容易把 conditional/unconditional 分裂成两套实现。
- 主要风险 3：`backbones/cond_unet.py` 和 `backbones/unet.py` 这种拆法，会把“是否有条件”固化成模型名，而不是输入协议。
- 结论：要定义 `batch` 和 `condition`，但要做成“薄协议”；不要定义一个万能 `BaseBackbone` 去统一所有生成方法。

## Key Changes
- 新建 `core/`，把基类和类型从根 `__init__.py` 移到 `core/module.py`、`core/types.py`；根 `__init__.py` 只做稳定导出。
- 数据代码只保留一处；推荐全部收敛到包内 `generative_models_lightning/data/`，不要同时保留根目录 `data_module/` 和包内 `datamodule/`。
- 定义 `GenerativeBatch`：只保留 `x`、`condition=None`、`target=None`、`metadata={}` 这些稳定字段，并提供 `to(device)`、`batch_size`、`as_model_kwargs()` 这类轻量方法。
- 定义 `ConditionBatch`：统一容纳 `labels`、`text`、`image`、`embeddings`、`extras` 这些可选分支；不要为每种条件类型做一套独立 batch 类。
- 规定 `unconditional == condition is None`；不要再用零标签、空张量或特殊 magic key 去伪装无条件。
- classifier-free guidance 的 dropout 或 blank-token 逻辑放在 `conditioning/` 层，不放在 dataset，也不散落在每个 backbone 里。
- 新建 `conditioning/`，其中 `types.py` 放 condition 容器，`encoders/` 放 label/text/image encoder，`adapters/` 负责把通用 condition 转成各方法需要的 kwargs。
- 扩散模块只接收规范化后的 condition/backbone kwargs，不再直接把 `y`、`s` 这种 ad-hoc 键名当公共协议。
- `backbones/` 不再承担“全方法共享完整模型”的职责；共享到全方法的只放 block，比如 ResBlock、CrossAttention、PatchEmbed、TimeEmbedding。
- 方法专用 backbone 放到各自目录，例如 `diffusion/backbones/denoisers/`、`flow/backbones/transforms/`、`vae/backbones/encoders_decoders/`、`gan/backbones/generators_discriminators/`。
- 如果以后某个 UNet 核心真的跨方法复用，抽成 `UNetCore`；外层再包 `UNetDenoiser`、`UNetDecoder` 这类方法语义适配器。

## Public Interfaces
- `dataset/datamodule` 统一返回 `GenerativeBatch`，不要返回裸 tuple 或随意 dict。
- `method module` 对外统一暴露 `compute_loss(batch)` 和 `sample(num_samples, condition=None, guidance=None, sampler=None)`。
- `backbone` 只接收“方法内规范化后的输入”，不直接消费原始 dataset 输出。
- diffusion 家族接口固定为 `forward(x_t, t, condition=None)`。
- flow 家族接口固定为 `forward(x, condition=None)` 和 `inverse(z, condition=None)`。
- VAE 家族接口固定为 `encode(x, condition=None)` 和 `decode(z, condition=None)`。
- GAN 家族接口固定为 `Generator.forward(z, condition=None)` 与 `Discriminator.forward(x, condition=None)`。
- 推荐目录骨架为：
```text
generative_models_lightning/
  core/
  conditioning/
  nn/
  data/
  diffusion/
  flow/
  vae/
  gan/
```

## Test Plan
- `GenerativeBatch`/`ConditionBatch` 单测覆盖 `.to(device)`、空 condition、label/text/image 混合 condition、序列化。
- 扩散单测覆盖 unconditional training/sample、label condition、text/image condition、CFG dropout。
- 接口单测保证每个方法族 backbone 都通过各自协议被 module 调用，不依赖裸 dict 键名。
- 回归单测保证 [gaussian_diffusion.py](d:/Projects/generative_models_lightining/generative_models_lightning/diffusion/process/gaussian_diffusion.py#L926) 这一类 unconditional 路径不再隐式依赖 `y`。
- 文档回归保证 [project-structure.md](d:/Projects/generative_models_lightining/docs/tutorials/project-structure.md#L20) 和真实目录一致，不再出现文档写 `base.py`、实现却在别处的漂移。

## Assumptions
- 目标是长期统一框架，而不是只服务一个 diffusion 实验。
- 近期条件类型至少包括 class label、image condition、text/context，所以需要统一 condition 容器，但不需要现在就为每个 modality 发明一套独立类型系统。
- 默认选择“通用 block + 方法专用 backbone”的组织方式，而不是“一个万能 BaseBackbone + 一堆可选 kwargs”。
- 默认选择“无条件是有条件接口的退化情况”，所以训练和采样都只保留一套主路径。
