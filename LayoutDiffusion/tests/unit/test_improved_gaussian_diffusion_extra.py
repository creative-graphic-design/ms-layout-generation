from __future__ import annotations

import numpy as np
import torch

from improved_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
)


class DummyModel(torch.nn.Module):
    def forward(self, x, t, **kwargs):
        return torch.zeros_like(x)


def test_gaussian_diffusion_sampling() -> None:
    betas = np.linspace(0.0001, 0.02, 100)
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )

    x_start = torch.randn(2, 3)
    t = torch.tensor([0, 1])
    noise = torch.randn_like(x_start)

    mean, var, log_var = diffusion.q_mean_variance(x_start, t)
    assert mean.shape == x_start.shape
    assert var.shape == x_start.shape
    assert log_var.shape == x_start.shape

    sample = diffusion.q_sample(x_start, t, noise=noise)
    assert sample.shape == x_start.shape

    posterior = diffusion.q_posterior_mean_variance(x_start, sample, t)
    assert posterior[0].shape == x_start.shape

    model = DummyModel()
    out = diffusion.p_mean_variance(model, sample, t)
    assert "mean" in out

    p_sample = diffusion.p_sample(model, sample, t)
    assert p_sample["sample"].shape == x_start.shape

    loop = diffusion.p_sample_loop(model, shape=(2, 3), device=x_start.device)
    assert loop.shape == (2, 3)

    ddim = diffusion.ddim_sample(model, sample, t)
    assert "sample" in ddim
