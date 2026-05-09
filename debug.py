
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from generative_models_lightning.diffusion.base_diffusion_module import BaseDiffusionLitModule
from lightning.pytorch.loggers import TensorBoardLogger
from generative_models_lightning.backbones.cond_unet import UNetModel
from generative_models_lightning.diffusion import SpacedDiffusion, space_timesteps
from generative_models_lightning.diffusion.utils import DiffusionMeanType, DiffusionVarType, DiffusionLossType
from generative_models_lightning.diffusion.beta_schedule import LinearBetaSchedule
from data_module.rodnet2021 import BaseDataModule, ConfMapConfig
from data_module.rodnet2021.rodnet2021_dataset import RODNet2021Dataset


if __name__ == "__main__":

    # set global random seed
    pl.seed_everything(42)

    torch.set_float32_matmul_precision("medium")  # (optional, to use Tensor Cores properly)

    module = BaseDiffusionLitModule(
        denoiser = UNetModel(
            image_size=128,
            in_channels=8,
            model_channels=128,
            out_channels=8,
            num_res_blocks=2,
            attention_resolutions=(16, 8),
            num_classes=3, 
            dropout=0.0,
            channel_mult=(1, 1, 2, 2, 4, 4),
            use_checkpoint=False,
            use_fp16=False,
            num_heads=4,
            num_head_channels=-1,
            num_heads_upsample=4,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
        ),
        scheduler=SpacedDiffusion(
            num_timesteps=1000,
            section_counts="1000",
            betas=LinearBetaSchedule(num_timesteps=1000),
            model_mean_type=DiffusionMeanType.EPSILON,
            model_var_type=DiffusionVarType.FIXED_LARGE,
            loss_type=DiffusionLossType.MSE,
            rescale_timesteps=False,
        ),
        mean_type=DiffusionMeanType.EPSILON,
        var_type=DiffusionVarType.FIXED_LARGE,
        loss_type=DiffusionLossType.MSE,

    )

    data_module = BaseDataModule(
        dataset=RODNet2021Dataset(
            rec_key='All',
            label_map_cfg=ConfMapConfig(),
        ),
        batch_size=4,
        num_workers=4,
        prepare_data_flag=False
    )
    

    logger = TensorBoardLogger("tb_logs", name="semantic_diffusion")
    
    # 或者在训练器配置中强制使用CPU
    trainer = pl.Trainer(
        accelerator="auto",  # 强制使用CPU
        devices='auto',
        max_epochs=80,
        # precision=args.precision,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        # deterministic=False,
        # strategy=None,
        enable_progress_bar=True,
    )

    # 开始训练
    trainer.fit(model=module, datamodule=data_module)

