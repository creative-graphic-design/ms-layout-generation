from types import SimpleNamespace

import pytest

from layoutformer_pp.trainer import get_trainer, MultiTaskTrainer, DSMultiTaskTrainer


def test_get_trainer_basic():
    args = SimpleNamespace(trainer="basic")
    assert get_trainer(args) is MultiTaskTrainer


def test_get_trainer_deepspeed():
    args = SimpleNamespace(trainer="deepspeed")
    assert get_trainer(args) is DSMultiTaskTrainer


def test_get_trainer_invalid():
    args = SimpleNamespace(trainer="unknown")
    with pytest.raises(NotImplementedError):
        get_trainer(args)
