from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from improved_diffusion import dist_util, distributed, resample, respace
from improved_diffusion import gaussian_diffusion as gd


def _ensure_dist_initialized() -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("DIFFUSION_NO_MPI", "1")
    dist_util.setup_dist()


def _cleanup_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def test_dist_util_setup_and_load(tmp_path) -> None:
    _ensure_dist_initialized()

    assert dist_util.dev().type in {"cpu", "cuda"}
    port = dist_util._find_free_port()
    assert isinstance(port, int)

    device = dist_util.dev()
    tensor = torch.tensor([1, 2, 3], device=device)
    path = tmp_path / "state.pt"
    torch.save({"t": tensor.cpu()}, path)
    loaded = dist_util.load_state_dict(str(path))
    assert loaded["t"].tolist() == [1, 2, 3]

    dist_util.sync_params([tensor])

    _cleanup_dist()


def test_distributed_helpers_without_init() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

    assert distributed.get_rank() == 0
    assert distributed.get_world_size() == 1
    assert distributed.is_primary()


def test_resample_and_respace() -> None:
    diffusion = SimpleNamespace(num_timesteps=5)
    sampler = resample.UniformSampler(diffusion)
    assert sampler.weights().shape == (5,)

    loss_sampler = resample.LossSecondMomentResampler(diffusion, history_per_term=2)
    loss_sampler.update_with_all_losses([0, 1], [0.1, 0.2])
    weights = loss_sampler.weights()
    assert weights.shape == (5,)

    steps = respace.space_timesteps(10, "ddim5")
    assert len(steps) == 5

    betas = np.linspace(0.1, 0.2, 5)
    spaced = respace.SpacedDiffusion(
        use_timesteps={0, 2, 4},
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE,
    )
    assert spaced.num_timesteps == 3
