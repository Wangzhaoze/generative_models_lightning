"""CycleGAN Lightning module adapted from the canonical PyTorch implementation."""

from __future__ import annotations

import itertools
from typing import Any

import torch
from torch import nn

from .base_gan_module import BaseTranslationGANModule
from .networks import GANLoss, init_gan_weights, make_patch_discriminator, make_resnet_generator

import random


class ImagePool:
    """Store previously generated images for more stable GAN training."""

    def __init__(self, pool_size: int) -> None:
        self.pool_size = int(pool_size)
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """Return a mix of new and buffered images."""
        if self.pool_size <= 0:
            return images

        return_images: list[torch.Tensor] = []
        for image in images:
            image = image.detach().unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
                continue

            if random.random() > 0.5:
                random_index = random.randint(0, self.pool_size - 1)
                buffered = self.images[random_index].clone()
                self.images[random_index] = image
                return_images.append(buffered)
            else:
                return_images.append(image)
        return torch.cat(return_images, dim=0)



class CycleGANModule(BaseTranslationGANModule):
    """Train unpaired image translation with cycle consistency."""

    def __init__(
        self,
        *,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        ndf: int = 64,
        n_blocks: int = 9,
        n_layers_D: int = 3,
        norm: str = "instance",
        use_dropout: bool = False,
        pool_size: int = 50,
        gan_mode: str = "lsgan",
        lambda_source: float = 10.0,
        lambda_target: float = 10.0,
        lambda_identity: float = 0.5,
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
        self.generator_source_to_target = make_resnet_generator(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            norm=norm,
            use_dropout=use_dropout,
            n_blocks=n_blocks,
        )
        self.generator_target_to_source = make_resnet_generator(
            input_nc=output_nc,
            output_nc=input_nc,
            ngf=ngf,
            norm=norm,
            use_dropout=use_dropout,
            n_blocks=n_blocks,
        )
        self.discriminator_target = make_patch_discriminator(
            input_nc=output_nc,
            ndf=ndf,
            n_layers=n_layers_D,
            norm=norm,
        )
        self.discriminator_source = make_patch_discriminator(
            input_nc=input_nc,
            ndf=ndf,
            n_layers=n_layers_D,
            norm=norm,
        )
        init_gan_weights(self.generator_source_to_target)
        init_gan_weights(self.generator_target_to_source)
        init_gan_weights(self.discriminator_target)
        init_gan_weights(self.discriminator_source)
        self.fake_source_pool = ImagePool(pool_size)
        self.fake_target_pool = ImagePool(pool_size)
        self.criterion_gan = GANLoss(gan_mode)
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.lambda_source = float(lambda_source)
        self.lambda_target = float(lambda_target)
        self.lambda_identity = float(lambda_identity)
        self.save_hyperparameters(
            ignore=(
                "generator_source_to_target",
                "generator_target_to_source",
                "discriminator_target",
                "discriminator_source",
                "fake_source_pool",
                "fake_target_pool",
                "criterion_gan",
                "criterion_cycle",
                "criterion_identity",
            )
        )

    def generator_parameters(self):
        return itertools.chain(
            self.generator_source_to_target.parameters(),
            self.generator_target_to_source.parameters(),
        )

    def discriminator_parameters(self):
        return itertools.chain(
            self.discriminator_source.parameters(),
            self.discriminator_target.parameters(),
        )

    def _normalize_direction(self, direction: str) -> str:
        mapping = {
            "source_to_target": "source_to_target",
            "a2b": "source_to_target",
            "target_to_source": "target_to_source",
            "b2a": "target_to_source",
        }
        if direction not in mapping:
            raise ValueError(f"Unsupported translation direction: {direction}")
        return mapping[direction]

    def translate(
        self,
        source: torch.Tensor,
        *,
        direction: str,
    ) -> torch.Tensor:
        resolved_direction = self._normalize_direction(direction)
        if resolved_direction == "source_to_target":
            return self.generator_source_to_target(source)
        return self.generator_target_to_source(source)

    def _discriminator_loss(
        self,
        discriminator: nn.Module,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> torch.Tensor:
        pred_real = discriminator(real_images)
        pred_fake = discriminator(fake_images.detach())
        loss_real = self.criterion_gan(pred_real, True)
        loss_fake = self.criterion_gan(pred_fake, False)
        return (loss_real + loss_fake) * 0.5

    def _generator_terms(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fake_target = self.generator_source_to_target(source)
        rec_source = self.generator_target_to_source(fake_target)
        fake_source = self.generator_target_to_source(target)
        rec_target = self.generator_source_to_target(fake_source)

        loss_g_source_to_target = self.criterion_gan(
            self.discriminator_target(fake_target),
            True,
        )
        loss_g_target_to_source = self.criterion_gan(
            self.discriminator_source(fake_source),
            True,
        )
        loss_cycle_source = (
            self.criterion_cycle(rec_source, source) * self.lambda_source
        )
        loss_cycle_target = (
            self.criterion_cycle(rec_target, target) * self.lambda_target
        )

        if self.lambda_identity > 0.0:
            identity_target = self.generator_source_to_target(target)
            identity_source = self.generator_target_to_source(source)
            loss_identity_target = (
                self.criterion_identity(identity_target, target)
                * self.lambda_target
                * self.lambda_identity
            )
            loss_identity_source = (
                self.criterion_identity(identity_source, source)
                * self.lambda_source
                * self.lambda_identity
            )
        else:
            loss_identity_target = torch.zeros((), device=source.device)
            loss_identity_source = torch.zeros((), device=source.device)

        loss_g = (
            loss_g_source_to_target
            + loss_g_target_to_source
            + loss_cycle_source
            + loss_cycle_target
            + loss_identity_target
            + loss_identity_source
        )
        return {
            "loss_g": loss_g,
            "loss_g_source_to_target": loss_g_source_to_target,
            "loss_g_target_to_source": loss_g_target_to_source,
            "loss_cycle_source": loss_cycle_source,
            "loss_cycle_target": loss_cycle_target,
            "loss_identity_source": loss_identity_source,
            "loss_identity_target": loss_identity_target,
            "fake_source": fake_source,
            "fake_target": fake_target,
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        source, target = self._unpack_batch(batch)
        d_optimizer, g_optimizer = self._optimizers()

        self.set_requires_grad(
            [self.discriminator_source, self.discriminator_target],
            False,
        )
        g_optimizer.zero_grad()
        generator_terms = self._generator_terms(source, target)
        self.manual_backward(generator_terms["loss_g"])
        g_optimizer.step()

        self.set_requires_grad(
            [self.discriminator_source, self.discriminator_target],
            True,
        )
        d_optimizer.zero_grad()
        loss_d_source = self._discriminator_loss(
            self.discriminator_source,
            source,
            self.fake_source_pool.query(generator_terms["fake_source"]),
        )
        loss_d_target = self._discriminator_loss(
            self.discriminator_target,
            target,
            self.fake_target_pool.query(generator_terms["fake_target"]),
        )
        loss_d = loss_d_source + loss_d_target
        self.manual_backward(loss_d)
        d_optimizer.step()

        self._log_terms(
            "train",
            {
                "loss_d": loss_d.detach(),
                "loss_d_source": loss_d_source.detach(),
                "loss_d_target": loss_d_target.detach(),
                **{
                    name: value.detach()
                    for name, value in generator_terms.items()
                    if name not in {"fake_source", "fake_target"}
                },
            },
            batch_size=source.shape[0],
            prog_bar_keys=("loss_d", "loss_g"),
        )
        return generator_terms["loss_g"].detach()

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        source, target = self._unpack_batch(batch)
        generator_terms = self._generator_terms(source, target)
        loss_d_source = self._discriminator_loss(
            self.discriminator_source,
            source,
            generator_terms["fake_source"],
        )
        loss_d_target = self._discriminator_loss(
            self.discriminator_target,
            target,
            generator_terms["fake_target"],
        )
        loss_d = loss_d_source + loss_d_target
        self._log_terms(
            "val",
            {
                "loss_d": loss_d,
                "loss_d_source": loss_d_source,
                "loss_d_target": loss_d_target,
                **{
                    name: value
                    for name, value in generator_terms.items()
                    if name not in {"fake_source", "fake_target"}
                },
            },
            batch_size=source.shape[0],
            prog_bar_keys=("loss_d", "loss_g"),
        )
        return generator_terms["loss_g"]
