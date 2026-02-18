from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import torch


def test_text_sample_bit_roundtrip() -> None:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "improved-diffusion"
        / "scripts"
        / "text_sample.py"
    )
    module = runpy.run_path(str(script_path))
    int2bit = module["int2bit"]
    bit2int = module["bit2int"]

    x = torch.tensor([1, 2, 3])
    bits = int2bit(x, n=3)
    restored = bit2int(bits)
    assert torch.equal(restored, x)


def test_batch_decode_script(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "improved-diffusion" / "scripts" / "batch_decode.py"

    model_dir = tmp_path / "model_dir"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "model_dummy.pt"
    model_file.write_text("x", encoding="utf-8")

    out_dir = repo_root / "results" / "generation_outputs" / "model_dir" / "type"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_base_name = f"model_dir.{model_file.name}"
    out_path = out_dir / f"{model_base_name}.samples_-1.0.json"
    out_path.write_text("[]", encoding="utf-8")

    old_argv = sys.argv
    old_cwd = os.getcwd()
    try:
        sys.argv = [
            str(script_path),
            str(model_dir),
            "-1.0",
            "model",
            "1",
            "1",
            "False",
            "-1",
            "type",
        ]
        os.chdir(repo_root / "improved-diffusion")
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    assert out_path.exists()
