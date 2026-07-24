from __future__ import annotations

import torch
from torch import nn

from generative_models_lightning.backbones.vae import AutoencoderKL
from generative_models_lightning.diffusion.gaussian_diffusion_module import (
    GaussianDiffusionModule,
)


def _tiny_vae(*, freeze: bool) -> AutoencoderKL:
    vae = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 8),
        layers_per_block=1,
        latent_channels=2,
        norm_num_groups=4,
        sample_size=8,
        mid_block_add_attention=False,
    )
    return vae.freeze() if freeze else vae


def test_klvae_encode_decode_and_forward_contract():
    vae = _tiny_vae(freeze=False)
    x = torch.randn(2, 1, 8, 8)

    posterior = vae.encode(x)
    latents = posterior.mode()
    reconstructed = vae.decode(latents)
    output = vae(x, sample_posterior=False)

    assert latents.shape == (2, 2, 4, 4)
    assert posterior.kl().shape[0] == 2
    assert reconstructed.shape == x.shape
    assert output.shape == x.shape


class _Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(2, 2, kernel_size=1)
        self.condition_shape = None

    def forward(self, x, timesteps, y):
        del timesteps
        self.condition_shape = tuple(y.shape)
        return self.projection(x) + y.mean(dim=1, keepdim=True) * 0.0


def test_ddpm_encodes_targets_and_resizes_condition():
    vae = _tiny_vae(freeze=True)
    denoiser = _Denoiser()
    module = GaussianDiffusionModule(
        denoiser=denoiser,
        vae=vae,
        freeze_vae=True,
        num_timesteps=4,
        beta_schedule="cosine",
        mean_type="epsilon",
        var_type="fixed_small",
        loss_type="mse",
        sample_shape=(2, 4, 4),
    )

    target = torch.randn(2, 1, 8, 8)
    condition = torch.randn(2, 1, 8, 8)
    terms = module.compute_loss_terms(
        target,
        {"cond": condition, "meta": {"ignored": True}},
    )

    assert terms["loss"].shape == (2,)
    assert terms["vae_kl"].shape[0] == 2
    assert denoiser.condition_shape == (2, 1, 4, 4)
    assert not any(parameter.requires_grad for parameter in vae.parameters())


def test_backbone_checkpoint_loading(tmp_path):
    trained = _tiny_vae(freeze=False)
    checkpoint_path = tmp_path / "klvae.ckpt"
    torch.save(
        {
            "state_dict": {
                f"vae.{key}": value for key, value in trained.state_dict().items()
            }
        },
        checkpoint_path,
    )
    loaded = _tiny_vae(freeze=False)
    _ = GaussianDiffusionModule(
        denoiser=_Denoiser(),
        vae=loaded,
        vae_checkpoint_path=str(checkpoint_path),
        vae_checkpoint_prefix="vae.",
        freeze_vae=True,
        num_timesteps=4,
        beta_schedule="cosine",
        mean_type="epsilon",
        var_type="fixed_small",
        loss_type="mse",
        sample_shape=(2, 4, 4),
    )

    expected = trained.state_dict()
    actual = loaded.state_dict()
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
