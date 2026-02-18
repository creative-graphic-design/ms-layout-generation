from types import SimpleNamespace

import torch
import pytest

from layoutformer_pp.model.layout_transformer.tokenizer import (
    LayoutTransformerTokenizer,
)
from layoutformer_pp.trainer.generator import Generator


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()


class DummyFid:
    def __init__(self):
        self.model = torch.nn.Identity()
        self.collected = []

    def collect_features(self, bboxes, labels, mask, real=False):
        self.collected.append((bboxes, labels, mask, real))

    def compute_score(self):
        return 0.0


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.seq_processor = None
        self.colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        self._task = None

    def switch_task(self, task):
        self._task = task

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"name": "sample"}


def _make_args(tmp_path):
    return SimpleNamespace(
        trainer="basic",
        backend="nccl",
        local_rank=0,
        eval_batch_size=1,
        out_dir=str(tmp_path),
        dataset="publaynet",
        bbox_format="ltrb",
        discrete_x_grid=10,
        discrete_y_grid=10,
        num_save=0,
    )


def test_generator_runs_on_cpu(tmp_path):
    args = _make_args(tmp_path)
    tokenizer = LayoutTransformerTokenizer(["label_1", "0", "1"])
    dataset = DummyDataset()
    fid = DummyFid()

    def test_step(model, data, seq_processor, tokenizer, device, constraint_fn=None):
        bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
        labels = torch.tensor([[1]])
        mask = torch.tensor([[True]])
        metrics = {
            "num_bbox_correct": 1.0,
            "num_bbox": 1.0,
            "num_label_correct": 1.0,
            "num_examples": 1.0,
        }
        out = {
            "gold_bboxes": bboxes,
            "gold_labels": labels,
            "pred_bboxes": bboxes,
            "pred_labels": labels,
            "mask": mask,
            "input_bboxes": bboxes,
            "input_labels": labels,
            "input_mask": mask,
            "pred_mask": mask,
            "gold_mask": mask,
        }
        return metrics, out

    generator = Generator(
        args,
        tokenizer,
        DummyModel(),
        None,
        dataset,
        fid,
        ds_ckpt_tag="unit",
        d2c_fn=lambda x: x,
        is_label_condition=False,
        saved_layouts=["pred"],
    )

    generator.switch_task("dummy", ["pred"])
    generator(test_step, draw_colors=dataset.colors)

    assert (tmp_path / "dummy" / "unit" / "metrics.pkl").exists()
    assert (tmp_path / "dummy" / "unit" / "results.pkl").exists()


def test_generator_saves_inputs_and_constraints(tmp_path):
    args = _make_args(tmp_path)
    args.num_save = 1
    tokenizer = LayoutTransformerTokenizer(["label_1", "0", "1"])
    dataset = DummyDataset()
    fid = DummyFid()
    constraint_used = {"value": False}

    def constraint_fn(*_args, **_kwargs):
        return None

    def test_step(model, data, seq_processor, tokenizer, device, constraint_fn=None):
        constraint_used["value"] = constraint_fn is not None
        bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
        labels = torch.tensor([[1]])
        mask = torch.tensor([[True]])
        metrics = {
            "num_bbox_correct": 1.0,
            "num_bbox": 1.0,
            "num_label_correct": 1.0,
            "num_examples": 1.0,
        }
        out = {
            "gold_bboxes": bboxes,
            "gold_labels": labels,
            "pred_bboxes": bboxes,
            "pred_labels": labels,
            "mask": mask,
            "input_bboxes": bboxes,
            "input_labels": labels,
            "input_mask": mask,
            "pred_mask": mask,
            "gold_mask": mask,
            "extra": ["extra"],
        }
        return metrics, out

    generator = Generator(
        args,
        tokenizer,
        DummyModel(),
        None,
        dataset,
        fid,
        ds_ckpt_tag="unit",
        d2c_fn=lambda x: x,
        is_label_condition=False,
        saved_layouts=["input", "gold", "pred"],
        save_entries=["extra"],
    )

    generator.switch_task("dummy", ["input", "gold", "pred"])
    generator(test_step, draw_colors=dataset.colors, constraint_fn=constraint_fn)

    assert constraint_used["value"] is True
    assert generator.results
    assert "input" in generator.results[0]
    assert "extra" in generator.results[0]


def test_generator_rico_overlap_and_violation_rate(tmp_path):
    args = _make_args(tmp_path)
    args.dataset = "rico"
    args.num_save = 1
    tokenizer = LayoutTransformerTokenizer(["label_1", "label_4", "0", "1"])
    dataset = DummyDataset()
    fid = DummyFid()

    def test_step(model, data, seq_processor, tokenizer, device, constraint_fn=None):
        bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]]])
        labels = torch.tensor([[4, 1]])
        mask = torch.tensor([[True, True]])
        metrics = {
            "num_bbox_correct": 1.0,
            "num_bbox": 1.0,
            "num_label_correct": 1.0,
            "num_examples": 1.0,
            "violation_num": 2.0,
            "rel_num": 4.0,
        }
        out = {
            "gold_bboxes": bboxes,
            "gold_labels": labels,
            "pred_bboxes": bboxes,
            "pred_labels": labels,
            "mask": mask,
            "pred_mask": mask,
            "gold_mask": mask,
        }
        return metrics, out

    generator = Generator(
        args,
        tokenizer,
        DummyModel(),
        None,
        dataset,
        fid,
        ds_ckpt_tag="unit",
        d2c_fn=lambda x: x,
        is_label_condition=False,
        saved_layouts=["pred"],
    )

    generator.switch_task("dummy", ["pred"])
    generator(test_step, draw_colors=dataset.colors)

    assert generator.violation_rate == pytest.approx(0.5)
