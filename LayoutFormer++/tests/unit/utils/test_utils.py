from types import SimpleNamespace

import torch

from layoutformer_pp.utils import utils


def test_save_and_load_arguments(tmp_path):
    data = {"alpha": 1, "beta": "two"}
    path = tmp_path / "args.json"

    utils.save_arguments(data, path)
    loaded = utils.load_arguments(path)

    assert loaded == data


def test_init_experiment_and_log_hparams(tmp_path):
    args = SimpleNamespace(
        seed=None,
        epoch=1,
        gradient_accumulation=2,
        batch_size=3,
        lr=0.1,
        max_num_elements=5,
        dataset="publaynet",
        clip_gradient=1.0,
    )

    out_dir = utils.init_experiment(args, tmp_path / "exp")

    assert out_dir.exists()
    assert args.seed is not None

    config = utils.log_hyperparameters(args, world_size=2)
    assert config["batch_size"] == 12
    assert config["dataset"] == "publaynet"


def test_collate_fn_requires_consistent_keys():
    batch = utils.collate_fn(
        [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]
    )

    assert set(batch.keys()) == {"a", "b"}
    assert batch["a"] == [1, 3]


def test_to_dense_batch_with_1d_and_2d():
    batch_1d = [torch.tensor([1, 2]), torch.tensor([3])]
    dense_1d, mask_1d = utils.to_dense_batch(batch_1d)

    assert dense_1d.shape == (2, 2)
    assert mask_1d.shape == (2, 2)
    assert mask_1d[1, 1].item() is False

    batch_2d = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])]
    dense_2d, mask_2d = utils.to_dense_batch(batch_2d)

    assert dense_2d.shape == (2, 2, 2)
    assert mask_2d.shape == (2, 2)


def test_parse_predicted_layout():
    labels, bboxes = utils.parse_predicted_layout("text 0 0 1 1 title 2 2 3 3")
    assert labels == ["text", "title"]
    assert bboxes == [[0, 0, 1, 1], [2, 2, 3, 3]]

    labels, bboxes = utils.parse_predicted_layout("invalid 0 0 1")
    assert labels is None
    assert bboxes is None
