"""WGAN-GP Lightning module adapted from the classic CIFAR reference code."""

from __future__ import annotations

from typing import Any

import torch

from .base_gan_module import BaseLatentGANModule
from .networks import (
    WGANGPCritic,
    WGANGPGenerator,
    calculate_gradient_penalty,
    init_gan_weights,
)


class WGANGPModule(BaseLatentGANModule):
    """Train a Wasserstein GAN with gradient penalty on image tensors."""

    def __init__(
        self,
        *,
        sample_shape: tuple[int, int, int] = (3, 32, 32),
        latent_dim: int = 128,
        model_dim: int = 64,
        lr: float = 1e-4,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        image_channels = int(sample_shape[0])
        super().__init__(
            sample_shape=sample_shape,
            latent_shape=(latent_dim,),
            lr=lr,
            weight_decay=weight_decay,
            generator_betas=(0.5, 0.9),
            discriminator_betas=(0.5, 0.9),
            n_critic=n_critic,
            **kwargs,
        )
        self.generator = WGANGPGenerator(
            latent_dim=latent_dim,
            dim=model_dim,
            out_channels=image_channels,
        )
        self.critic = WGANGPCritic(
            in_channels=image_channels,
            dim=model_dim,
        )
        init_gan_weights(self.generator)
        init_gan_weights(self.critic)
        self.lambda_gp = float(lambda_gp)
        self.save_hyperparameters(ignore=("generator", "critic"))

    def generator_parameters(self):
        return self.generator.parameters()

    def discriminator_parameters(self):
        return self.critic.parameters()

    def forward_generator(
        self,
        latents: torch.Tensor,
        cond: Any | None = None,
    ) -> torch.Tensor:
        del cond
        return self.generator(latents)

    def _critic_terms(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        real_score = self.critic(real_images)
        fake_score = self.critic(fake_images.detach())
        gradient_penalty = calculate_gradient_penalty(
            self.critic,
            real_images,
            fake_images.detach(),
            lambda_gp=self.lambda_gp,
        )
        wasserstein_distance = real_score.mean() - fake_score.mean()
        loss_d = fake_score.mean() - real_score.mean() + gradient_penalty
        return {
            "loss_d": loss_d,
            "wasserstein_distance": wasserstein_distance,
            "gradient_penalty": gradient_penalty,
        }

    def _generator_loss(self, fake_images: torch.Tensor) -> torch.Tensor:
        return -self.critic(fake_images).mean()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        real_images, _ = self._unpack_batch(batch)
        d_optimizer, g_optimizer = self._optimizers()

        fake_images = self.forward_generator(self.sample_latents(real_images.shape[0]))
        d_optimizer.zero_grad()
        critic_terms = self._critic_terms(real_images, fake_images)
        self.manual_backward(critic_terms["loss_d"])
        d_optimizer.step()

        should_update_generator = ((batch_idx + 1) % self.n_critic) == 0
        if should_update_generator:
            g_optimizer.zero_grad()
            generator_images = self.forward_generator(
                self.sample_latents(real_images.shape[0])
            )
            loss_g = self._generator_loss(generator_images)
            self.manual_backward(loss_g)
            g_optimizer.step()
        else:
            with torch.no_grad():
                loss_g = self._generator_loss(
                    self.forward_generator(self.sample_latents(real_images.shape[0]))
                )

        self._log_terms(
            "train",
            {
                **{name: value.detach() for name, value in critic_terms.items()},
                "loss_g": loss_g.detach(),
            },
            batch_size=real_images.shape[0],
            prog_bar_keys=("loss_d", "loss_g"),
        )
        return loss_g.detach()

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        real_images, _ = self._unpack_batch(batch)
        with torch.enable_grad():
            fake_images = self.forward_generator(
                self.sample_latents(real_images.shape[0])
            )
            critic_terms = self._critic_terms(real_images, fake_images)
        loss_g = self._generator_loss(fake_images)
        self._log_terms(
            "val",
            {
                **critic_terms,
                "loss_g": loss_g,
            },
            batch_size=real_images.shape[0],
            prog_bar_keys=("loss_d", "loss_g"),
        )
        return loss_g
