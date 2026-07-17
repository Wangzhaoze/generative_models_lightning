import torch
from torch import nn

from generative_models_lightning.flow_model.Flow_matching.meanflow import MeanFlow
from generative_models_lightning.flow_model.Flow_matching.path import MeanFlowProbPath
from generative_models_lightning.flow_model.Flow_matching.solver import MeanFlowEulerSolver


class _TinyMeanFlowModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, t, r, y=None):
        output = self.scale * x
        if y is not None and y.ndim == x.ndim:
            output = output + 0.01 * y[:, :1]
        return output


def test_meanflow_path_uses_reverse_conditional_ot() -> None:
    path = MeanFlowProbPath(flow_ratio=0.5, time_dist=("uniform", 0.0, 1.0))
    noise = torch.full((2, 1, 2, 2), 2.0)
    data = torch.zeros_like(noise)
    time = torch.tensor([0.0, 0.25])

    sample = path.sample(x_0=noise, x_1=data, t=time)

    torch.testing.assert_close(sample.x_t[0], data[0])
    torch.testing.assert_close(sample.x_t[1], torch.full_like(data[1], 0.5))
    torch.testing.assert_close(sample.dx_t, noise - data)


def test_meanflow_euler_solver_uses_descending_time_pairs() -> None:
    def constant_velocity(x, t, r):
        return torch.ones_like(x) * 2.0

    solver = MeanFlowEulerSolver(constant_velocity)
    result = solver.sample(
        x_init=torch.zeros(2, 1, 2, 2),
        time_grid=torch.tensor([1.0, 0.5, 0.0]),
    )

    torch.testing.assert_close(result, torch.full_like(result, -2.0))


def test_composed_meanflow_supports_dense_condition_and_backward() -> None:
    model = _TinyMeanFlowModel()
    meanflow = MeanFlow(
        channels=1,
        image_size=(3, 4),
        num_classes=None,
        normalizer=("mean_std", 0.0, 1.0),
        time_dist=("uniform", 0.0, 1.0),
        cfg_ratio=0.0,
        cfg_scale=None,
        valid_data_min=-0.99,
    )
    data = torch.rand(2, 1, 3, 4)
    condition = torch.rand(2, 2, 3, 4)

    loss, mse = meanflow.loss(model, data, condition)
    loss.backward()
    samples = meanflow.sample(
        model,
        cond=condition,
        sample_steps=2,
        device="cpu",
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(mse)
    assert model.scale.grad is not None
    assert samples.shape == data.shape
