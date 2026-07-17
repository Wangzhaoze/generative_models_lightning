from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from generative_models_lightning import BaseGenerativeModule


FlowTerms = Mapping[str, torch.Tensor]


class BaseFlowModule(BaseGenerativeModule, ABC):
    """Base Lightning module for path-based generative flow models.

    ``path`` and ``solver`` are injected dependencies.  They may be existing
    implementations from ``Flow_matching``, MeanFlow processes, or objects
    instantiated from configuration.  Subclasses only adapt their
    algorithm-specific path objective through :meth:`path_step`.
    """

    def __init__(
        self,
        model: nn.Module,
        path: Any,
        solver: Any,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, **kwargs)
        self.model = model
        self.path = path
        self.solver = solver

    def sample_source(self, reference: torch.Tensor) -> torch.Tensor:
        """Sample the default Gaussian source distribution."""
        return torch.randn_like(reference)

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample the default continuous flow time in ``[0, 1]``."""
        return torch.rand(batch_size, device=device)

    @abstractmethod
    def path_step(self, batch: Any, batch_idx: int) -> FlowTerms:
        """Use ``self.path`` to build the algorithm-specific training terms.

        The returned mapping must contain ``loss``.  Flow Matching subclasses
        can call ``self.path.sample(...)`` here, while MeanFlow subclasses can
        call their process' JVP objective.
        """

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        terms = self.path_step(batch, batch_idx)
        if "loss" not in terms:
            raise ValueError("path_step must return a mapping containing 'loss'.")

        loss = terms["loss"].mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite flow training loss: {loss.detach().item()}")

        for name, value in terms.items():
            if torch.is_tensor(value):
                self.log(
                    "train_loss" if name == "loss" else f"train_{name}",
                    value.mean(),
                    prog_bar=name == "loss",
                    sync_dist=True,
                )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        terms = self.path_step(batch, batch_idx)
        if "loss" not in terms:
            raise ValueError("path_step must return a mapping containing 'loss'.")

        loss = terms["loss"].mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite flow validation loss: {loss.detach().item()}")

        for name, value in terms.items():
            if torch.is_tensor(value):
                self.log(
                    "val_loss" if name == "loss" else f"val_{name}",
                    value.mean(),
                    prog_bar=name == "loss",
                    sync_dist=True,
                )
        return loss

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        return super().predict_step(batch, batch_idx)

    @torch.inference_mode()
    def generate(self, *solver_args: Any, **solver_kwargs: Any) -> Any:
        """Generate samples with the injected solver.

        Solver-specific arguments are forwarded unchanged, so changing the
        solver does not require changing this base class.
        """
        if self.solver is None:
            raise RuntimeError("A solver is required for generation.")
        return self.solver.sample(*solver_args, **solver_kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
