"""DCGAN Lightning module adapted from the official PyTorch example."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base_gan_module import BaseLatentGANModule
from .networks import DCGANDiscriminator, DCGANGenerator, init_dcgan_weights


class DCGANModule(BaseLatentGANModule):
    """Train a classic DCGAN with BCE adversarial losses."""

    def __init__(
        self,
        *,
        sample_shape: tuple[int, int, int] = (3, 32, 32),
        latent_dim: int = 100,
        generator_channels: int = 64,
        discriminator_channels: int = 64,
        beta1: float = 0.5,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        image_channels, image_size, image_size_2 = sample_shape
        if image_size != image_size_2:
            raise ValueError("DCGAN sample_shape must be square")
        super().__init__(
            sample_shape=sample_shape,
            latent_shape=(latent_dim, 1, 1),
            lr=lr,
            weight_decay=weight_decay,
            generator_betas=(beta1, 0.999),
            discriminator_betas=(beta1, 0.999),
            **kwargs,
        )
        self.generator = DCGANGenerator(
            latent_channels=latent_dim,
            ngf=generator_channels,
            out_channels=image_channels,
            image_size=image_size,
        )
        self.discriminator = DCGANDiscriminator(
            in_channels=image_channels,
            ndf=discriminator_channels,
            image_size=image_size,
        )
        init_dcgan_weights(self.generator)
        init_dcgan_weights(self.discriminator)
        self.criterion = nn.BCELoss()
        self.save_hyperparameters(ignore=("generator", "discriminator", "criterion"))

    def generator_parameters(self):
        return self.generator.parameters()

    def discriminator_parameters(self):
        return self.discriminator.parameters()

    def forward_generator(
        self,
        latents: torch.Tensor,
        cond: Any | None = None,
    ) -> torch.Tensor:
        del cond
        return self.generator(latents)

    def _generator_loss(self, fake_images: torch.Tensor) -> torch.Tensor:
        targets = torch.ones(fake_images.shape[0], device=fake_images.device)
        predictions = self.discriminator(fake_images)
        return self.criterion(predictions, targets)

    def _discriminator_loss(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_targets = torch.ones(real_images.shape[0], device=real_images.device)
        fake_targets = torch.zeros(real_images.shape[0], device=real_images.device)
        real_predictions = self.discriminator(real_images)
        fake_predictions = self.discriminator(fake_images.detach())
        loss_real = self.criterion(real_predictions, real_targets)
        loss_fake = self.criterion(fake_predictions, fake_targets)
        return (loss_real + loss_fake), loss_real, loss_fake

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        real_images, _ = self._unpack_batch(batch)
        d_optimizer, g_optimizer = self._optimizers()
        fake_images = self.forward_generator(self.sample_latents(real_images.shape[0]))

        d_optimizer.zero_grad()
        loss_d, loss_d_real, loss_d_fake = self._discriminator_loss(
            real_images,
            fake_images,
        )
        self.manual_backward(loss_d)
        d_optimizer.step()

        g_optimizer.zero_grad()
        refreshed_fake_images = self.forward_generator(
            self.sample_latents(real_images.shape[0])
        )
        loss_g = self._generator_loss(refreshed_fake_images)
        self.manual_backward(loss_g)
        g_optimizer.step()

        self._log_terms(
            "train",
            {
                "loss_d": loss_d.detach(),
                "loss_d_real": loss_d_real.detach(),
                "loss_d_fake": loss_d_fake.detach(),
                "loss_g": loss_g.detach(),
            },
            batch_size=real_images.shape[0],
            prog_bar_keys=("loss_d", "loss_g"),
        )
        return loss_g.detach()

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        real_images, _ = self._unpack_batch(batch)
        fake_images = self.forward_generator(self.sample_latents(real_images.shape[0]))
        loss_d, loss_d_real, loss_d_fake = self._discriminator_loss(
            real_images,
            fake_images,
        )
        loss_g = self._generator_loss(fake_images)
        self._log_terms(
            "val",
            {
                "loss_d": loss_d,
                "loss_d_real": loss_d_real,
                "loss_d_fake": loss_d_fake,
                "loss_g": loss_g,
            },
            batch_size=real_images.shape[0],
            prog_bar_keys=("loss_d", "loss_g"),
        )
        return loss_g
