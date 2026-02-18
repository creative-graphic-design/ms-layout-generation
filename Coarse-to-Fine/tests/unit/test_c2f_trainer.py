"""Unit tests for c2f_trainer.py Trainer class (GPU/DDP, no mocks)."""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from layoutformer_pp.trainer.utils import CheckpointMeasurement
from coarse_to_fine.c2f_trainer import Trainer, linear


class TinyDataset(Dataset):
    def __init__(self, size=8):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "x": torch.randn(4),
            "name": f"sample_{idx}",
        }


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)


def collate_fn(batch):
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "name": [item["name"] for item in batch],
    }


def train_step(args, model, data, device):
    x = data["x"].to(device)
    out = model(x)
    loss = out.mean()
    return {
        "group_bounding_box": loss,
        "label_in_one_group": loss,
        "grouped_box": loss,
        "grouped_label": loss,
        "KL": loss,
    }


def eval_step(model, data, device):
    batch_size = data["x"].size(0)
    seq_len = 2

    bboxes = torch.zeros(batch_size, seq_len, 4, dtype=torch.long, device=device)
    labels = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    masks = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    ori = {
        "bboxes": bboxes,
        "labels": labels,
        "masks": masks,
    }
    rec = {
        "bboxes": bboxes.clone(),
        "labels": labels.clone(),
    }
    mask_info = {
        "ori_box_mask": masks,
        "rec_box_mask": masks.clone(),
    }
    return ori, rec, mask_info


class TestLinearFunction:
    def test_linear_at_min(self):
        result = linear(0.0, 1.0, 0, 0, 100)
        assert result == 0.0

    def test_linear_at_max(self):
        result = linear(0.0, 1.0, 100, 0, 100)
        assert result == 1.0

    def test_linear_at_middle(self):
        result = linear(0.0, 1.0, 50, 0, 100)
        assert result == 0.5

    def test_linear_below_min(self):
        result = linear(0.0, 1.0, -10, 0, 100)
        assert result == 0.0

    def test_linear_above_max(self):
        result = linear(0.0, 1.0, 150, 0, 100)
        assert result == 1.0

    def test_linear_different_range(self):
        result = linear(10.0, 20.0, 50, 0, 100)
        assert result == 15.0


@pytest.mark.gpu
@pytest.mark.unit
def test_trainer_ddp_single_gpu(gpu_available, ddp_env):
    if not gpu_available:
        pytest.skip("GPU not available")

    if dist.is_initialized():
        dist.destroy_process_group()

    backend = "nccl" if dist.is_nccl_available() else "gloo"

    with tempfile.TemporaryDirectory() as tmpdir:
        args = SimpleNamespace(
            backend=backend,
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            out_dir=tmpdir,
            max_num_elements=4,
            gradient_accumulation=1,
            enable_clip_gradient=True,
            clip_gradient=1.0,
            batch_size=2,
            eval_batch_size=2,
            epoch=1,
            train_log_step=1,
            bbox_format="ltrb",
            discrete_x_grid=8,
            discrete_y_grid=8,
            kl_start_step=0,
            kl_end_step=10,
            seed=0,
            lr=1e-3,
            dataset="rico",
        )

        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        dataset = TinyDataset(size=4)

        trainer = Trainer(
            task_name="c2f_test",
            args=args,
            model=model,
            train_dataset=dataset,
            val_dataset=dataset,
            optimizer=optimizer,
            checkpoint_measure=CheckpointMeasurement.OVERLAP,
            collate_fn=collate_fn,
            is_debug=True,
            task_config={},
        )

        trainer(train_step, eval_step)

        assert os.path.exists(os.path.join(tmpdir, "checkpoint.pth.tar"))
        assert os.path.exists(os.path.join(tmpdir, "model_best.pth.tar"))
        assert os.path.exists(os.path.join(tmpdir, "val_output.pkl"))

        trainer.clean_up()
        assert not dist.is_initialized()
