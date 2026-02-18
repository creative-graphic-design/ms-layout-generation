from __future__ import annotations

import numpy as np
import torch as th

from improved_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)


def test_q_sample_matches_expected_mean_for_zero_noise() -> None:
    betas = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_LARGE,
        loss_type=LossType.MSE,
    )
    x_start = th.tensor([[1.0, -1.0], [0.5, 0.0]])
    t = th.tensor([0, 1], dtype=th.long)
    noise = th.zeros_like(x_start)

    sample = diffusion.q_sample(x_start, t, noise=noise)

    sqrt_alphas = th.from_numpy(diffusion.sqrt_alphas_cumprod).to(x_start)
    expected = sqrt_alphas[t].unsqueeze(-1) * x_start
    assert th.allclose(sample, expected, atol=1e-6)


def test_q_mean_variance_shapes_and_values() -> None:
    betas = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_LARGE,
        loss_type=LossType.MSE,
    )
    x_start = th.tensor([[1.0, -1.0], [0.5, 0.0]])
    t = th.tensor([0, 2], dtype=th.long)

    mean, variance, log_variance = diffusion.q_mean_variance(x_start, t)

    assert mean.shape == x_start.shape
    assert variance.shape == x_start.shape
    assert log_variance.shape == x_start.shape

    sqrt_alphas = th.from_numpy(diffusion.sqrt_alphas_cumprod).to(x_start)
    expected_mean = sqrt_alphas[t].unsqueeze(-1) * x_start
    assert th.allclose(mean, expected_mean, atol=1e-6)
