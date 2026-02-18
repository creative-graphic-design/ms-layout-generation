from __future__ import annotations

import argparse
from types import SimpleNamespace
from pathlib import Path

import torch

from eval_src.utils import os_utils, utils


def test_makedirs_and_files_exist(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir"
    os_utils.makedirs(str(target))
    assert target.exists()

    file_a = target / "a.txt"
    file_a.write_text("x", encoding="utf-8")
    assert os_utils.files_exist([str(file_a)])


def test_save_and_load_arguments(tmp_path: Path) -> None:
    args = {"a": 1, "b": "c"}
    path = tmp_path / "args.json"
    utils.save_arguments(args, str(path))
    loaded = utils.load_arguments(str(path))
    assert loaded == args


def test_init_experiment_sets_seed(tmp_path: Path) -> None:
    args = SimpleNamespace(seed=None)
    out_dir = utils.init_experiment(args, str(tmp_path / "out"))
    assert out_dir.exists()
    assert (out_dir / "args.json").exists()


def test_log_hyperparameters() -> None:
    args = SimpleNamespace(
        seed=1,
        epoch=2,
        gradient_accumulation=1,
        batch_size=2,
        lr=0.01,
        pretrained_lm_name="t5",
        max_num_elements=20,
        dataset="rico",
        clip_gradient=5,
    )
    config = utils.log_hyperparameters(args, world_size=1)
    assert config["batch_size"] == 2
    assert config["dataset"] == "rico"


def test_collate_and_dense_batch() -> None:
    batch = [{"a": torch.tensor([1, 2])}, {"a": torch.tensor([3])}]
    out = utils.collate_fn(batch)
    assert "a" in out
    dense, mask = utils.to_dense_batch(out["a"])
    assert dense.shape[0] == 2
    assert mask.shape[0] == 2


def test_parse_predicted_layout() -> None:
    labels, bbox = utils.parse_predicted_layout("Text 0 0 10 10 Image 1 1 2 2")
    assert labels == ["Text", "Image"]
    assert bbox == [[0, 0, 10, 10], [1, 1, 2, 2]]


def test_add_arguments_parser() -> None:
    parser = argparse.ArgumentParser()
    from eval_src.utils import config

    config.add_arguments(parser)
    args = parser.parse_args([])
    assert hasattr(args, "dataset")
    assert hasattr(args, "batch_size")
