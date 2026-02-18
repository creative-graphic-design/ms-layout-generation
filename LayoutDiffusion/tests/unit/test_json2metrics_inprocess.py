from __future__ import annotations

import os
import pickle
import runpy
import shutil
import sys
from pathlib import Path

import torch


def _write_rico_pt(path: Path, layouts: list[dict[str, torch.Tensor]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(layouts, path)


def test_json2metrics_runs_inprocess(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fid_src = repo_root / "eval_src" / "net" / "fid_rico.pth.tar"
    fid_dest = tmp_path / "eval_src" / "net" / "fid_rico.pth.tar"
    fid_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fid_src, fid_dest)

    data_dir = tmp_path / "data" / "raw_datasets" / "rico" / "pre_processed_20_25"
    layouts = [
        {
            "bboxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
        },
        {
            "bboxes": torch.tensor([[5, 5, 8, 8]], dtype=torch.float32),
            "labels": torch.tensor([2], dtype=torch.long),
        },
    ]
    _write_rico_pt(data_dir / "test.pt", layouts)
    _write_rico_pt(data_dir / "val.pt", layouts)

    pred_dir = tmp_path / "results" / "generation_outputs" / "tests"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / "samples.json"
    pred_lines = [
        '["START Text 0 0 10 10 END"]',
        '["START Image 1 5 5 10 10 END"]',
    ]
    pred_path.write_text("\n".join(pred_lines) + "\n", encoding="utf-8")

    old_cwd = os.getcwd()
    old_argv = sys.argv
    try:
        os.chdir(tmp_path)
        sys.argv = ["json2metrics", str(pred_path)]
        runpy.run_module("layout_diffusion.json2metrics", run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    processed_path = pred_dir / "processed.pt"
    assert processed_path.exists()

    with processed_path.open("rb") as handle:
        processed = pickle.load(handle)

    assert processed
    assert "pred" in processed[0]
