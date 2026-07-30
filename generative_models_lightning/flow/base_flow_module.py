"""Algorithm-agnostic Lightning scaffolding for continuous flow models."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from generative_models_lightning import BaseGenerativeModule


FlowTerms = Mapping[str, torch.Tensor]


class BaseFlowModule(BaseGenerativeModule, ABC):
    """Share Lightning orchestration without coupling flow mathematics."""

    def __init__(
        self,
        model: torch.nn.Module,
        sample_shape: Sequence[int] | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, **kwargs)
        self.model = model
        self.sample_shape = (
            tuple(int(value) for value in sample_shape)
            if sample_shape is not None
            else None
        )
        self.warmup_steps = int(warmup_steps)
        if self.sample_shape is not None and any(
            value <= 0 for value in self.sample_shape
        ):
            raise ValueError("sample_shape dimensions must be greater than zero")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, Any | None]:
        """Normalize supported batch forms to ``(target, condition)``."""
        if isinstance(batch, Mapping):
            if "x" not in batch:
                raise KeyError("Flow batch mapping must contain key 'x'")
            x = batch["x"]
            cond = batch.get("cond")
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, condition = batch
            cond = condition.get("cond") if isinstance(condition, Mapping) else condition
        elif isinstance(batch, torch.Tensor):
            x, cond = batch, None
        else:
            raise TypeError("Batch must be a tensor, a mapping, or an (x, cond) pair")

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Batch input x must be a Tensor, got {type(x).__name__}")
        return x, cond

    def sample_source(self, reference: torch.Tensor) -> torch.Tensor:
        """Sample the default Gaussian source distribution."""
        return torch.randn_like(reference)

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample a single continuous time in ``[0, 1]`` per example."""
        return torch.rand(batch_size, device=device)

    def sample_time_pair(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an upper/lower time pair.

        Continuous Flow Matching subclasses can ignore the lower time by
        inheriting this default ``(t, t)`` implementation.
        """
        t = self.sample_time(batch_size=batch_size, device=device)
        return t, t

    def apply_condition_dropout(
        self,
        cond: Any | None,
        drop_rate: float,
    ) -> Any | None:
        """Randomly zero whole-example conditions for CFG-style training."""
        if cond is None or drop_rate <= 0.0:
            return cond
        if not isinstance(cond, torch.Tensor):
            return cond
        if cond.shape[0] == 0:
            return cond

        mask = torch.rand(cond.shape[0], device=cond.device) < drop_rate
        if not torch.any(mask):
            return cond
        dropped = cond.clone()
        dropped[mask] = 0
        return dropped

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        x, cond = self._unpack_batch(batch)
        loss_terms = self.compute_loss_terms(x=x, cond=cond)
        if "loss" not in loss_terms:
            raise ValueError(
                "compute_loss_terms() must return a dictionary with key 'loss'"
            )

        loss = loss_terms["loss"].mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {stage} loss: {loss.detach().item()}")

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )
        for name, value in loss_terms.items():
            if name == "loss":
                continue
            self.log(
                f"{stage}/{name}",
                value.mean(),
                on_step=stage == "train",
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, stage="val")

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        del batch_idx, dataloader_idx
        return self.generate(batch=batch)

    @abstractmethod
    def compute_loss_terms(
        self,
        x: torch.Tensor,
        cond: Any | None,
    ) -> dict[str, torch.Tensor]:
        """Compute one or more algorithm-specific loss terms."""
        raise NotImplementedError

    @torch.inference_mode()
    @abstractmethod
    def sample(
        self,
        shape: Sequence[int],
        cond: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate samples from noise using an algorithm-specific solver."""
        raise NotImplementedError

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
        """Generate from an input batch, an explicit shape, or ``sample_shape``."""
        batch_x = None
        batch_cond = None
        if batch is not None:
            batch_x, batch_cond = self._unpack_batch(batch)
        if cond is None:
            cond = batch_cond
        if isinstance(cond, Mapping):
            cond = cond.get("cond")

        if shape is None:
            if batch_x is not None:
                shape = tuple(batch_x.shape)
            elif self.sample_shape is not None:
                shape = (int(batch_size or 1), *self.sample_shape)
            else:
                raise ValueError("Provide batch, shape, or configure sample_shape")

        return self.sample(
            shape=tuple(int(value) for value in shape),
            cond=cond,
            **kwargs,
        )

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if self.warmup_steps == 0:
            return optimizer

        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
        if total_steps <= 0:
            return optimizer

        warmup_steps = self.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
