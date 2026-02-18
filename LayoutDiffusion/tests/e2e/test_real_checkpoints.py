from __future__ import annotations

import pytest
import torch


@pytest.mark.real_data
@pytest.mark.slow
def test_load_real_checkpoint_rico(real_checkpoint_rico) -> None:
    state = torch.load(real_checkpoint_rico, map_location="cpu")
    assert state is not None


@pytest.mark.real_data
@pytest.mark.slow
def test_load_real_checkpoint_pub(real_checkpoint_pub) -> None:
    state = torch.load(real_checkpoint_pub, map_location="cpu")
    assert state is not None
