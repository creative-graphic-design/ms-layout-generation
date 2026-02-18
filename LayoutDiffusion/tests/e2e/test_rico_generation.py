from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _prepare_checkpoint(
    tmp_path: Path,
    source_dir: Path,
    dataset_dir: Path,
    *,
    diffusion_steps: int | None = None,
) -> Path:
    target_dir = tmp_path / "checkpoint"
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "ema_0.9999_175000.pt",
        "random_emb.torch",
        "vocab.json",
        "training_args.json",
    ]:
        shutil.copy2(source_dir / filename, target_dir / filename)

    training_args_path = target_dir / "training_args.json"
    training_args = json.loads(training_args_path.read_text(encoding="utf-8"))
    training_args["e2e_train"] = str(dataset_dir)
    training_args["checkpoint_path"] = str(target_dir)
    if diffusion_steps is not None:
        training_args["diffusion_steps"] = diffusion_steps
    training_args_path.write_text(json.dumps(training_args, indent=2), encoding="utf-8")
    return target_dir


@pytest.mark.e2e
def test_rico_unconditional_generation(
    tmp_path: Path,
    layoutdiffusion_data_root: Path,
    pythonpath_env: dict[str, str],
) -> None:
    source_dir = (
        layoutdiffusion_data_root
        / "results"
        / "checkpoint"
        / "discrete_gaussian_pow2.5_aux_lex_ltrb_200_fine_4e5"
    )
    dataset_dir = (
        layoutdiffusion_data_root / "data" / "processed_datasets" / "RICO_ltrb_lex"
    )
    checkpoint_dir = _prepare_checkpoint(
        tmp_path, source_dir, dataset_dir, diffusion_steps=20
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/text_sample.py",
        "--model_path",
        str(checkpoint_dir / "ema_0.9999_175000.pt"),
        "--batch_size",
        "1",
        "--num_samples",
        "1",
        "--top_p",
        "-1.0",
        "--out_dir",
        str(out_dir),
        "--multistep",
        "False",
        "--clamp",
        "none",
        "--verbose",
        "yes",
    ]

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        cmd,
        cwd=repo_root / "improved-diffusion",
        env=pythonpath_env,
        check=True,
    )

    assert list(out_dir.glob("*.npz"))


@pytest.mark.e2e
def test_rico_type_conditioned_generation(
    tmp_path: Path,
    layoutdiffusion_data_root: Path,
    pythonpath_env: dict[str, str],
) -> None:
    source_dir = (
        layoutdiffusion_data_root
        / "results"
        / "checkpoint"
        / "discrete_gaussian_pow2.5_aux_lex_ltrb_200_fine_4e5"
    )
    dataset_dir = (
        layoutdiffusion_data_root / "data" / "processed_datasets" / "RICO_ltrb_lex"
    )
    checkpoint_dir = _prepare_checkpoint(
        tmp_path, source_dir, dataset_dir, diffusion_steps=160
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/text_sample.py",
        "--model_path",
        str(checkpoint_dir / "ema_0.9999_175000.pt"),
        "--batch_size",
        "1",
        "--num_samples",
        "1",
        "--top_p",
        "-1.0",
        "--out_dir",
        str(out_dir),
        "--multistep",
        "False",
        "--clamp",
        "none",
        "--verbose",
        "yes",
        "--constrained",
        "type",
    ]

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        cmd,
        cwd=repo_root / "improved-diffusion",
        env=pythonpath_env,
        check=True,
    )

    assert list(out_dir.glob("*.npz"))
