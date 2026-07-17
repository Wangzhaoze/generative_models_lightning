# Adapted from https://github.com/haidog-yaqub/MeanFlow
# Commit: 18b67b25a3af86d199005023bb190024d28233e7
# Original license: MIT. See the vendored MeanFlow LICENSE.

from collections.abc import Sequence

import torch
from torch import Tensor

from .affine import AffineProbPath
from .scheduler.scheduler import ConvexScheduler, SchedulerOutput


class _MeanFlowScheduler(ConvexScheduler):
    """Reverse conditional-OT schedule used by MeanFlow.

    With ``x_0`` as noise and ``x_1`` as normalized data, this schedule gives
    ``x_t = t * x_0 + (1 - t) * x_1``.  MeanFlow therefore runs from noise at
    ``t=1`` to data at ``t=0``.
    """

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=1 - t,
            sigma_t=t,
            d_alpha_t=-torch.ones_like(t),
            d_sigma_t=torch.ones_like(t),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return 1 - kappa


class MeanFlowProbPath(AffineProbPath):
    """Affine probability path and paired-time sampler for MeanFlow.

    ``sample`` is inherited from :class:`AffineProbPath` and returns the
    standard :class:`PathSample`.  Its inputs have the following meaning:

    - ``x_0``: Gaussian noise.
    - ``x_1``: normalized data.
    - ``t``: MeanFlow time, where 1 is noise and 0 is data.

    The returned ``dx_t`` is ``x_0 - x_1``, the velocity used by the MeanFlow
    objective.  The second time ``r`` belongs to that objective and is sampled
    separately with :meth:`sample_t_r`.
    """

    def __init__(
        self,
        flow_ratio: float = 0.5,
        time_dist: Sequence[float | str] = ("lognorm", -0.4, 1.0),
    ) -> None:
        if not 0.0 <= flow_ratio <= 1.0:
            raise ValueError("flow_ratio must be in [0, 1].")
        if not time_dist:
            raise ValueError("time_dist must contain a distribution name.")

        distribution = time_dist[0]
        if distribution not in {"uniform", "lognorm"}:
            raise ValueError(f"unsupported time distribution: {distribution}")
        if distribution == "lognorm" and len(time_dist) < 3:
            raise ValueError("lognorm time_dist must provide mu and sigma.")

        super().__init__(scheduler=_MeanFlowScheduler())
        self.flow_ratio = float(flow_ratio)
        self.time_dist = tuple(time_dist)

    def sample_t_r(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> tuple[Tensor, Tensor]:
        """Sample paired MeanFlow times with ``r <= t``.

        A fraction ``flow_ratio`` of the batch is changed to instantaneous
        flow training by setting ``r = t``.
        """

        distribution = self.time_dist[0]
        if distribution == "uniform":
            samples = torch.rand(batch_size, 2, device=device)
        else:
            mu, sigma = self.time_dist[-2:]
            samples = torch.randn(batch_size, 2, device=device)
            samples = torch.sigmoid(samples * float(sigma) + float(mu))

        t = torch.maximum(samples[:, 0], samples[:, 1])
        r = torch.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        if num_selected:
            indices = torch.randperm(batch_size, device=device)[:num_selected]
            r = r.clone()
            r[indices] = t[indices]

        return t, r
