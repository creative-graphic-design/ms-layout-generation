"""Unit tests for c2f_generator.py Generator class (GPU/DDP, no mocks)."""

import os
import tempfile
import multiprocessing as mp
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from coarse_to_fine.c2f_generator import Generator


mp.set_start_method("spawn", force=True)


class TinyDataset(Dataset):
    def __init__(self, size=2):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "name": f"sample_{idx}",
        }


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)


class TinyFID:
    def __init__(self):
        self.model = torch.nn.Identity()
        self.calls = 0

    def collect_features(self, *args, **kwargs):
        self.calls += 1

    def compute_score(self):
        return 0.0


class ColorMap:
    def __getitem__(self, key):
        if torch.is_tensor(key):
            key = int(key.item())
        return (255, 0, 0) if key else (255, 255, 255)


def collate_fn(batch):
    return {"name": [item["name"] for item in batch]}


def run_step(args, model, data, device):
    batch_size = len(data["name"])
    seq_len = 2
    group_len = 2

    bboxes = torch.zeros(batch_size, seq_len, 4, dtype=torch.long, device=device)
    labels = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    group_bboxes = torch.zeros(batch_size, group_len, 4, dtype=torch.long, device=device)
    group_labels = torch.ones(batch_size, group_len, dtype=torch.long, device=device)

    ori = {
        "bboxes": bboxes,
        "labels": labels,
    }
    out = {
        "bboxes": bboxes.clone(),
        "labels": labels.clone(),
        "group_bounding_box": group_bboxes,
        "label_in_one_group": group_labels,
    }
    masks = {
        "ori_box_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "gen_box_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "gen_group_bounding_box_mask": torch.ones(batch_size, group_len, dtype=torch.bool),
    }
    return ori, out, masks


def run_step_with_extra(args, model, data, device):
    ori, out, masks = run_step(args, model, data, device)
    out["extra"] = torch.zeros(len(data["name"]))
    return ori, out, masks


@pytest.mark.gpu
@pytest.mark.unit
def test_generator_ddp_single_gpu(gpu_available, ddp_env):
    if not gpu_available:
        pytest.skip("GPU not available")

    if dist.is_initialized():
        dist.destroy_process_group()

    backend = "nccl" if dist.is_nccl_available() else "gloo"

    with tempfile.TemporaryDirectory() as tmpdir:
        args = SimpleNamespace(
            backend=backend,
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            trainer="ddp",
            out_dir=tmpdir,
            eval_batch_size=2,
            discrete_x_grid=8,
            discrete_y_grid=8,
            bbox_format="ltrb",
            dataset="rico",
            num_save=0,
        )

        generator = Generator(
            args=args,
            model=TinyModel(),
            test_dataset=TinyDataset(size=2),
            fid_model=TinyFID(),
            collate_fn=collate_fn,
        )

        draw_colors = {i: (255, 0, 0) for i in range(10)}
        generator(run_step, draw_colors)

        assert os.path.exists(os.path.join(tmpdir, "metrics.pkl"))
        assert os.path.exists(os.path.join(tmpdir, "results.pkl"))

        generator.clean_up()
        assert not dist.is_initialized()


@pytest.mark.gpu
@pytest.mark.unit
def test_generator_single_checkpoint_and_images(gpu_available):
    if not gpu_available:
        pytest.skip("GPU not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        args = SimpleNamespace(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            local_rank=0,
            trainer="single",
            out_dir=tmpdir,
            eval_batch_size=1,
            discrete_x_grid=8,
            discrete_y_grid=8,
            bbox_format="xywh",
            dataset="rico",
            num_save=1,
        )

        model = TinyModel()
        state_dict = model.state_dict()
        ckpt_state = {
            "module.linear.weight": state_dict["linear.weight"],
            "module.linear.bias": state_dict["linear.bias"],
        }
        ckpt_path = os.path.join(tmpdir, "checkpoint.pth.tar")
        torch.save(ckpt_state, ckpt_path)

        generator = Generator(
            args=args,
            model=model,
            test_dataset=TinyDataset(size=1),
            fid_model=TinyFID(),
            ckpt_path=ckpt_path,
            save_entries=["extra"],
            collate_fn=collate_fn,
        )

        generator(run_step_with_extra, ColorMap())

        assert os.path.exists(os.path.join(tmpdir, "metrics.pkl"))
        assert os.path.exists(os.path.join(tmpdir, "results.pkl"))
        assert os.path.exists(os.path.join(tmpdir, "pics"))


@pytest.mark.gpu
@pytest.mark.unit
def test_generator_collect_layouts_ltwh(gpu_available):
    if not gpu_available:
        pytest.skip("GPU not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        args = SimpleNamespace(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            local_rank=0,
            trainer="single",
            out_dir=tmpdir,
            eval_batch_size=1,
            discrete_x_grid=8,
            discrete_y_grid=8,
            bbox_format="ltwh",
            dataset="rico",
            num_save=0,
        )

        generator = Generator(
            args=args,
            model=TinyModel(),
            test_dataset=TinyDataset(size=1),
            fid_model=TinyFID(),
            collate_fn=collate_fn,
        )

        bboxes = torch.zeros(1, 2, 4, dtype=torch.long, device=generator.device)
        labels = torch.ones(1, 2, dtype=torch.long, device=generator.device)
        mask = torch.tensor([[True, False]])
        layouts, _ = generator.collect_layouts(bboxes, labels, mask, [])
        assert len(layouts) == 1
