"""Shared Lightning scaffolding for adversarial generative models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Iterable, Sequence
from typing import Any

import torch

from generative_models_lightning import BaseGenerativeModule


class BaseGANModule(BaseGenerativeModule, ABC):
    """Thin base class for GAN training orchestration and public interfaces."""

    def __init__(
        self,
        *,
        sample_shape: Sequence[int] | None = None,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        generator_lr: float | None = None,
        discriminator_lr: float | None = None,
        generator_betas: tuple[float, float] = (0.5, 0.999),
        discriminator_betas: tuple[float, float] = (0.5, 0.999),
        n_critic: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, **kwargs)
        self.automatic_optimization = False
        self.sample_shape = (
            tuple(int(value) for value in sample_shape)
            if sample_shape is not None
            else None
        )
        self.generator_lr = float(generator_lr if generator_lr is not None else lr)
        self.discriminator_lr = float(
            discriminator_lr if discriminator_lr is not None else lr
        )
        self.generator_betas = tuple(float(value) for value in generator_betas)
        self.discriminator_betas = tuple(
            float(value) for value in discriminator_betas
        )
        self.n_critic = int(n_critic)
        if self.sample_shape is not None and any(
            value <= 0 for value in self.sample_shape
        ):
            raise ValueError("sample_shape dimensions must be greater than zero")
        if self.n_critic <= 0:
            raise ValueError("n_critic must be greater than zero")

    @abstractmethod
    def generator_parameters(self) -> Iterable[torch.nn.Parameter]:
        raise NotImplementedError

    @abstractmethod
    def discriminator_parameters(self) -> Iterable[torch.nn.Parameter]:
        raise NotImplementedError

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        d_optimizer = torch.optim.Adam(
            self.discriminator_parameters(),
            lr=self.discriminator_lr,
            betas=self.discriminator_betas,
            weight_decay=self.weight_decay,
        )
        g_optimizer = torch.optim.Adam(
            self.generator_parameters(),
            lr=self.generator_lr,
            betas=self.generator_betas,
            weight_decay=self.weight_decay,
        )
        return [d_optimizer, g_optimizer]

    def _optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)) or len(optimizers) != 2:
            raise RuntimeError("GAN modules expect exactly two optimizers")
        return optimizers[0], optimizers[1]

    def _mean_loss(self, value: torch.Tensor) -> torch.Tensor:
        return value.mean() if value.ndim > 0 else value

    def _log_terms(
        self,
        stage: str,
        terms: dict[str, torch.Tensor | float],
        *,
        batch_size: int,
        prog_bar_keys: Sequence[str] = (),
    ) -> None:
        for name, value in terms.items():
            tensor = (
                value
                if isinstance(value, torch.Tensor)
                else torch.tensor(float(value), device=self.device)
            )
            self.log(
                f"{stage}/{name}",
                self._mean_loss(tensor),
                prog_bar=name in prog_bar_keys,
                on_step=stage == "train",
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    @staticmethod
    def set_requires_grad(
        modules: torch.nn.Module | Sequence[torch.nn.Module],
        requires_grad: bool,
    ) -> None:
        if isinstance(modules, torch.nn.Module):
            modules = [modules]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(requires_grad)

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        del batch_idx, dataloader_idx
        return self.generate(batch=batch)


"""Base helpers for latent-to-image GANs such as DCGAN and WGAN-GP."""





class BaseLatentGANModule(BaseGANModule, ABC):
    """Shared public interface for latent-image GAN families."""

    def __init__(
        self,
        *,
        sample_shape: Sequence[int],
        latent_shape: Sequence[int] | None = None,
        latent_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        if latent_shape is None:
            if latent_dim is None:
                raise ValueError("Provide latent_shape or latent_dim")
            latent_shape = (int(latent_dim), 1, 1)
        self.latent_shape = tuple(int(value) for value in latent_shape)
        if any(value <= 0 for value in self.latent_shape):
            raise ValueError("latent_shape dimensions must be greater than zero")
        super().__init__(sample_shape=sample_shape, **kwargs)

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, Any | None]:
        if isinstance(batch, Mapping):
            if "x" not in batch:
                raise KeyError("Latent GAN batch mapping must contain key 'x'")
            x = batch["x"]
            cond = batch.get("cond")
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, cond = batch
        elif isinstance(batch, torch.Tensor):
            x, cond = batch, None
        else:
            raise TypeError("Batch must be a tensor, mapping, or an (x, cond) pair")
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Batch input x must be a Tensor, got {type(x).__name__}")
        return x, cond

    def sample_latents(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.randn(
            (int(batch_size), *self.latent_shape),
            device=self.device if device is None else device,
            dtype=torch.float32,
            generator=generator,
        )

    @abstractmethod
    def forward_generator(
        self,
        latents: torch.Tensor,
        cond: Any | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def sample(
        self,
        shape: Sequence[int],
        cond: Any | None = None,
        *,
        latents: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.sample_shape is None:
            raise ValueError("Latent GANs require sample_shape to be configured")
        resolved_shape = tuple(int(value) for value in shape)
        if tuple(resolved_shape[1:]) != tuple(self.sample_shape):
            raise ValueError(
                "Requested shape must match configured sample_shape: "
                f"{resolved_shape[1:]} != {self.sample_shape}"
            )
        if latents is None:
            latents = self.sample_latents(
                resolved_shape[0],
                device=self.device,
                generator=generator,
            )
        samples = self.forward_generator(latents, cond=cond)
        if samples.shape != torch.Size(resolved_shape):
            raise ValueError(
                "Generator output shape does not match requested shape: "
                f"{tuple(samples.shape)} != {resolved_shape}"
            )
        return samples

    @torch.inference_mode()
    def generate(
        self,
        batch: Any | None = None,
        *,
        batch_size: int | None = None,
        shape: Sequence[int] | None = None,
        cond: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch_x = None
        batch_cond = None
        if batch is not None:
            batch_x, batch_cond = self._unpack_batch(batch)
        if cond is None:
            cond = batch_cond
        if shape is None:
            if batch_x is not None:
                shape = tuple(batch_x.shape)
            elif self.sample_shape is not None:
                shape = (int(batch_size or 1), *self.sample_shape)
            else:
                raise ValueError("Provide batch, shape, or configure sample_shape")
        return self.sample(shape=shape, cond=cond, **kwargs)



class BaseTranslationGANModule(BaseGANModule, ABC):
    """Shared public interface for paired and unpaired translation models."""

    def __init__(
        self,
        *,
        default_direction: str = "source_to_target",
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_shape=None, **kwargs)
        self.default_direction = default_direction

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, Mapping):
            if "source" not in batch or "target" not in batch:
                raise KeyError("Translation batch mapping must contain source and target")
            source = batch["source"]
            target = batch["target"]
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            source, target = batch
        else:
            raise TypeError(
                "Batch must be a mapping with source/target or a (source, target) pair"
            )
        if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("source and target batch values must both be Tensors")
        return source, target

    @abstractmethod
    def translate(
        self,
        source: torch.Tensor,
        *,
        direction: str,
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def generate(
        self,
        batch: Any | None = None,
        *,
        source: torch.Tensor | None = None,
        direction: str | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if source is None:
            if batch is None:
                raise ValueError("Provide batch or source for translation generation")
            source, _ = self._unpack_batch(batch)
        return self.translate(
            source,
            direction=self.default_direction if direction is None else direction,
        )
