"""Unit tests for coarse_to_fine.main utilities."""

import os
from types import SimpleNamespace

import pytest
import argparse
import torch

from layoutformer_pp.data.base import RicoDataset, PubLayNetDataset
from coarse_to_fine.main import (
    add_sos_and_eos,
    rel_to_abs,
    train_step,
    eval_step,
    inference_step,
    create_dataset,
    add_task_arguments,
    add_trainer_arguments,
)


class DummyC2FModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, data, device, feed_gt=True):
        n, g, s, _ = data["grouped_bboxes"].shape
        d_box = max(self.args.discrete_x_grid, self.args.discrete_y_grid)

        rec = {
            "group_bounding_box": torch.randn(n, g + 2, 4, d_box, device=device),
            "label_in_one_group": torch.randn(n, g + 2, self.args.num_labels + 2, device=device),
            "grouped_bboxes": torch.randn(n, g, s, 4, d_box, device=device),
            "grouped_labels": torch.randn(n, g, s, self.args.num_labels + 3, device=device),
        }
        kl_info = {
            "mu": torch.zeros(n, 1, device=device),
            "logvar": torch.zeros(n, 1, device=device),
            "z": torch.zeros(n, 1, device=device),
        }
        return data, rec, kl_info

    def inference(self, device):
        n = 1
        g = 4
        s = 3
        d_box = max(self.args.discrete_x_grid, self.args.discrete_y_grid)

        label_in_one_group = torch.zeros(n, g, self.args.num_labels + 2, device=device)
        label_in_one_group[0, 0, 0] = 1.0
        label_in_one_group[0, 1, self.args.num_labels + 1] = 1.0

        grouped_labels = torch.zeros(n, g, s, self.args.num_labels + 3, device=device)
        grouped_labels[0, 0, 0, 1] = 1.0
        grouped_labels[0, 0, 1, self.args.num_labels + 2] = 1.0

        grouped_bboxes = torch.zeros(n, g, s, 4, d_box, device=device)
        group_bounding_box = torch.zeros(n, g, 4, d_box, device=device)

        return {
            "group_bounding_box": group_bounding_box,
            "label_in_one_group": label_in_one_group,
            "grouped_bboxes": grouped_bboxes,
            "grouped_labels": grouped_labels,
        }


def make_padding_input():
    return {
        "labels": [torch.tensor([1, 2, 3])],
        "bboxes": [torch.tensor([[0.1, 0.1, 0.2, 0.2],
                                 [0.3, 0.3, 0.2, 0.2],
                                 [0.5, 0.5, 0.2, 0.2]])],
        "label_in_one_group": [torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0, 1.0]])],
        "group_bounding_box": [torch.tensor([[0.0, 0.0, 1.0, 1.0],
                                             [0.0, 0.0, 1.0, 1.0],
                                             [0.0, 0.0, 1.0, 1.0]])],
        "grouped_label": [[torch.tensor([1, 2, 3])]],
        "grouped_box": [[torch.tensor([[0.1, 0.1, 0.2, 0.2],
                                       [0.3, 0.3, 0.2, 0.2],
                                       [0.5, 0.5, 0.2, 0.2]])]],
    }


def test_add_sos_and_eos():
    data = {
        "discrete_gold_bboxes": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([1]),
    }
    group_info = {
        "group_bounding_box": torch.tensor([[1, 1, 2, 2]]),
        "label_in_one_group": torch.tensor([[1.0, 0.0, 0.0]]),
        "grouped_label": [torch.tensor([1])],
        "grouped_box": [torch.tensor([[1, 1, 1, 1]])],
    }
    sos, eos = 10, 11
    out = add_sos_and_eos(data, group_info, sos, eos)

    assert out["labels"][0].item() == sos
    assert out["labels"][-1].item() == eos
    assert out["label_in_one_group"][0][0].item() == 1
    assert out["label_in_one_group"][-1][-1].item() == 1


def test_rel_to_abs():
    class Args:
        discrete_x_grid = 10
        discrete_y_grid = 10

    elements_boxes = [[1, 2, 3, 4]]
    group_box = [0, 0, 10, 20]
    out = rel_to_abs(elements_boxes, group_box, Args)

    assert out[0] == [1, 4, 3, 8]


def test_train_and_eval_step():
    device = torch.device("cpu")
    data = make_padding_input()
    args = SimpleNamespace(
        num_labels=3,
        discrete_x_grid=8,
        discrete_y_grid=8,
        group_box_weight=1.0,
        group_label_weight=1.0,
        box_weight=1.0,
        label_weight=1.0,
        kl_weight=1.0,
    )
    model = DummyC2FModel(args)

    loss = train_step(args, model, data, device)
    assert set(loss.keys()) == {
        "group_bounding_box",
        "label_in_one_group",
        "grouped_box",
        "grouped_label",
        "KL",
    }

    ori, rec, masks = eval_step(model, data, device)
    assert "bboxes" in rec
    assert "ori_box_mask" in masks


def test_inference_step():
    device = torch.device("cpu")
    data = make_padding_input()
    args = SimpleNamespace(
        num_labels=3,
        discrete_x_grid=8,
        discrete_y_grid=8,
    )
    model = DummyC2FModel(args)

    ori, gen, masks = inference_step(args, model, data, device)

    assert "bboxes" in gen
    assert "labels" in gen
    assert "gen_box_mask" in masks


def _write_preprocessed(root, data_name, max_num_elements, label_count, sample):
    base = os.path.join(root, data_name)
    pre_dir = os.path.join(base, f"pre_processed_{max_num_elements}_{label_count}")
    os.makedirs(pre_dir, exist_ok=True)
    for split in ("train", "val", "test"):
        torch.save([sample], os.path.join(pre_dir, f"{split}.pt"))


def test_create_dataset_with_preprocessed(tmp_path):
    sample = {
        "bboxes": torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2]]),
        "labels": torch.tensor([1, 2]),
        "name": "sample_0",
    }
    max_num_elements = 8

    _write_preprocessed(
        tmp_path,
        "rico",
        max_num_elements,
        len(RicoDataset.labels),
        sample,
    )
    _write_preprocessed(
        tmp_path,
        "publaynet",
        max_num_elements,
        len(PubLayNetDataset.labels),
        sample,
    )

    args = SimpleNamespace(
        dataset="rico",
        data_dir=str(tmp_path),
        max_num_elements=max_num_elements,
        num_labels=5,
        discrete_x_grid=8,
        discrete_y_grid=8,
    )
    rico_ds = create_dataset(args, split="train")
    item = rico_ds[0]
    assert "group_bounding_box" in item
    assert "grouped_box" in item
    assert "grouped_label" in item

    args.dataset = "publaynet"
    pub_ds = create_dataset(args, split="train")
    item = pub_ds[0]
    assert "group_bounding_box" in item


def test_create_dataset_invalid():
    args = SimpleNamespace(dataset="unknown", data_dir="/tmp")
    with pytest.raises(NotImplementedError):
        create_dataset(args, split="train")


def test_add_argument_helpers():
    parser = argparse.ArgumentParser()
    add_task_arguments(parser)
    add_trainer_arguments(parser)
    args = parser.parse_args([])
    assert args.num_labels == 25
    assert args.trainer == "basic"
