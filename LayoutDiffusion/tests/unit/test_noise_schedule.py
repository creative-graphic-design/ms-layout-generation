from __future__ import annotations

import numpy as np
import pytest

from improved_diffusion.discrete_diffusion import alpha_schedule
from improved_diffusion.gaussian_diffusion import get_named_beta_schedule


@pytest.mark.parametrize(
    "schedule_name",
    [
        "linear",
        "cosine",
        "sqrt",
        "mix_sqrt",
        "trunc_cos",
        "trunc_lin",
        "pw_lin",
    ],
)
def test_get_named_beta_schedule_range(schedule_name: str) -> None:
    betas = get_named_beta_schedule(schedule_name, 20)
    assert betas.shape == (20,)
    assert np.all(np.isfinite(betas))
    assert np.all(betas > 0)
    assert betas.max() <= 1.5


def test_get_named_beta_schedule_unknown() -> None:
    with pytest.raises(NotImplementedError):
        get_named_beta_schedule("not-a-schedule", 10)


@pytest.mark.parametrize("matrix_policy", [1, 2])
def test_alpha_schedule_shapes_and_ranges(matrix_policy: int) -> None:
    time_step = 10
    outputs = alpha_schedule(time_step, matrix_policy=matrix_policy, type_classes=25)
    assert len(outputs) == 12
    at, at1, bt1, bt2, ct, ct1, att, att1, btt1, btt2, ctt, ctt1 = outputs
    for arr in (at, at1, bt1, bt2, ct, ct1):
        assert len(arr) >= time_step
        assert np.all(np.isfinite(arr))
    for arr in (att, att1, btt1, btt2, ctt, ctt1):
        assert len(arr) >= time_step
        assert np.all(np.isfinite(arr))
