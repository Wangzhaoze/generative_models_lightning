"""pix2pix Lightning module adapted from the canonical PyTorch implementation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base_gan_module import BaseTranslationGANModule
from .networks import GANLoss, init_gan_weights, make_patch_discriminator, make_unet_generator


class Pix2PixModule(BaseTranslationGANModule):
    """Train paired image-to-image translation with a PatchGAN discriminator."""

    def __init__(
        self,
        *,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        ndf: int = 64,
        netG: str = "unet_256",
        n_layers_D: int = 3,
        norm: str = "batch",
        use_dropout: bool = False,
        lambda_l1: float = 100.0,
        beta1: float = 0.5,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            generator_betas=(beta1, 0.999),
            discriminator_betas=(beta1, 0.999),
            **kwargs,
        )
        self.generator = make_unet_generator(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            norm=norm,
            use_dropout=use_dropout,
            netG=netG,
        )
        self.discriminator = make_patch_discriminator(
            input_nc=input_nc + output_nc,
            ndf=ndf,
            n_layers=n_layers_D,
            norm=norm,
        )
        init_gan_weights(self.generator)
        init_gan_weights(self.discriminator)
        self.criterion_gan = GANLoss("vanilla")
        self.criterion_l1 = nn.L1Loss()
        self.lambda_l1 = float(lambda_l1)
        self.save_hyperparameters(
            ignore=("generator", "discriminator", "criterion_gan", "criterion_l1")
        )

    def generator_parameters(self):
        return self.generator.parameters()

    def discriminator_parameters(self):
        return self.discriminator.parameters()

    def translate(
        self,
        source: torch.Tensor,
        *,
        direction: str,
    ) -> torch.Tensor:
        if direction != "source_to_target":
            raise ValueError("Pix2PixModule only supports source_to_target translation")
        return self.generator(source)

    def _discriminator_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        fake_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_fake = self.discriminator(torch.cat((source, fake_target.detach()), dim=1))
        pred_real = self.discriminator(torch.cat((source, target), dim=1))
        loss_fake = self.criterion_gan(pred_fake, False)
        loss_real = self.criterion_gan(pred_real, True)
        return (loss_fake + loss_real) * 0.5, loss_real, loss_fake

    def _generator_terms(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        fake_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred_fake = self.discriminator(torch.cat((source, fake_target), dim=1))
        loss_g_gan = self.criterion_gan(pred_fake, True)
        loss_g_l1 = self.criterion_l1(fake_target, target) * self.lambda_l1
        return {
            "loss_g": loss_g_gan + loss_g_l1,
            "loss_g_gan": loss_g_gan,
            "loss_g_l1": loss_g_l1,
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        source, target = self._unpack_batch(batch)
        d_optimizer, g_optimizer = self._optimizers()
        fake_target = self.translate(source, direction="source_to_target")

        self.set_requires_grad(self.discriminator, True)
        d_optimizer.zero_grad()
        loss_d, loss_d_real, loss_d_fake = self._discriminator_loss(
            source,
            target,
            fake_target,
        )
        self.manual_backward(loss_d)
        d_optimizer.step()

        self.set_requires_grad(self.discriminator, False)
        g_optimizer.zero_grad()
        generator_terms = self._generator_terms(source, target, fake_target)
        self.manual_backward(generator_terms["loss_g"])
        g_optimizer.step()
        self.set_requires_grad(self.discriminator, True)

        self._log_terms(
            "train",
            {
                "loss_d": loss_d.detach(),
                "loss_d_real": loss_d_real.detach(),
                "loss_d_fake": loss_d_fake.detach(),
                **{name: value.detach() for name, value in generator_terms.items()},
            },
            batch_size=source.shape[0],
            prog_bar_keys=("loss_d", "loss_g"),
        )
        return generator_terms["loss_g"].detach()

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        source, target = self._unpack_batch(batch)
        fake_target = self.translate(source, direction="source_to_target")
        loss_d, loss_d_real, loss_d_fake = self._discriminator_loss(
            source,
            target,
            fake_target,
        )
        generator_terms = self._generator_terms(source, target, fake_target)
        self._log_terms(
            "val",
            {
                "loss_d": loss_d,
                "loss_d_real": loss_d_real,
                "loss_d_fake": loss_d_fake,
                **generator_terms,
            },
            batch_size=source.shape[0],
            prog_bar_keys=("loss_d", "loss_g"),
        )
        return generator_terms["loss_g"]
