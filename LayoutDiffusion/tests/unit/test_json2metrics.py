from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import torch


def _write_rico_pt(path: Path, layouts: list[dict[str, torch.Tensor]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(layouts, path)


def test_json2metrics_generates_processed_output(
    tmp_path: Path, repo_root: Path, pythonpath_env: dict[str, str]
) -> None:
    eval_src_link = tmp_path / "eval_src"
    if not eval_src_link.exists():
        eval_src_link.symlink_to(repo_root / "eval_src", target_is_directory=True)

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

    env = pythonpath_env.copy()
    env.setdefault("PYTHONHASHSEED", "0")

    result = subprocess.run(
        [sys.executable, "-m", "layout_diffusion.json2metrics", str(pred_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "miou" in result.stdout

    processed_path = pred_dir / "processed.pt"
    assert processed_path.exists()

    with processed_path.open("rb") as handle:
        processed = pickle.load(handle)

    assert isinstance(processed, list)
    assert processed
    assert "pred" in processed[0]
    assert len(processed[0]["pred"]) == 2
