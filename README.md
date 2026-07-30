# <div align="center">**Generative Models Lightning**</div>

<p align="center">
    <a href="https://wangzhaoze.github.io/">Zhaoze Wang</a><sup>1</sup>,
    <a href="">Chenlin Lang</a><sup>2</sup>
</p>


<p align="center" style="font-size: 0.9em; font-style: italic;">
  <sup>1</sup> Brandenburg University of Technology Cottbus-Senftenberg,
  <sup>2</sup> Leibniz University Hannover 
</p>
  
<div align=center>
    <img src="https://img.shields.io/badge/Python-3.10.16-3776AB.svg?style=for-the-badge&logo=python" alt="python">
    <img src=https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C.svg?style=for-the-badge&logo=pytorch>
    <img src=https://img.shields.io/badge/Lightning-2.5.0-purple?style=for-the-badge&logo=lightning>
</div>

<div align=center>
    <img src="https://img.shields.io/badge/Docker-gray?style=for-the-badge&logo=docker&logoColor=white&labelColor=%23007FFF" alt="Docker">
    <img src="https://img.shields.io/badge/Wandb-gray?style=for-the-badge&logo=weightsandbiases" alt="wandb">
</div>

## Diffusion Structure

All diffusion families share the algorithm-agnostic
`BaseDiffusionModule`. Each algorithm then keeps its complete mathematical
process and Lightning integration in one file:

```text
generative_models_lightning/diffusion/
├── base_diffusion_module.py  # batch, logging, generate
├── gaussian_diffusion.py     # DDPM/DDIM, schedules, spacing, Lightning
└── edm.py                    # preconditioning, loss, sampler, EMA, Lightning
```

Training is always launched through the same Hydra + Lightning entry point.
Each standalone config defines the model, data module, callbacks, logger, and
trainer without adding another training script.

## Conditional EDM

The `edm_exp` experiment provides NVLabs-style EDM preconditioning, weighted
denoising loss, Euler-Heun sampling, EMA, and classifier-free guidance.

Run an offline smoke training job:

```powershell
conda run -n dl python train.py --config-name edm_cifar10 `
  data_module.use_fake_data=true `
  data_module.batch_size=2 `
  data_module.num_workers=0 `
  trainer.fast_dev_run=true
```

Train on CIFAR-10 with the default configuration:

```powershell
conda run -n dl python train.py --config-name edm_cifar10
```

Train the conditional Gaussian diffusion baseline with the same data and
backbone:

```powershell
conda run -n dl python train.py --config-name gaussian_cifar10
```

The CIFAR-10 images are normalized to `[-1, 1]`. Conditions remain compact
one-hot vectors until the backbone adapter expands them for the existing
spatially conditioned U-Net.
