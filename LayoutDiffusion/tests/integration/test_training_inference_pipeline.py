from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _write_small_dataset(
    source_dir: Path,
    dest_dir: Path,
    *,
    train_lines: int = 2,
    valid_lines: int = 1,
    test_lines: int = 1,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": train_lines,
        "valid": valid_lines,
        "test": test_lines,
    }
    for split, count in splits.items():
        src_file = source_dir / f"src1_{split}.txt"
        lines = src_file.read_text(encoding="utf-8").splitlines()
        snippet = lines[:count]
        dest_file = dest_dir / f"src1_{split}.txt"
        dest_file.write_text("\n".join(snippet) + "\n", encoding="utf-8")


@pytest.mark.integration
def test_training_and_inference_pipeline(
    tmp_path: Path,
    layoutdiffusion_data_root: Path,
    pythonpath_env: dict[str, str],
) -> None:
    source_dataset = (
        layoutdiffusion_data_root / "data" / "processed_datasets" / "RICO_ltrb_lex"
    )
    dataset_dir = tmp_path / "dataset"
    _write_small_dataset(
        source_dataset, dataset_dir, train_lines=2, valid_lines=1, test_lines=1
    )

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        "scripts/train.py",
        "--checkpoint_path",
        str(checkpoint_dir),
        "--e2e_train",
        str(dataset_dir),
        "--model_arch",
        "transformer",
        "--modality",
        "e2e-tgt",
        "--training_mode",
        "discrete1",
        "--experiment",
        "random",
        "--experiment_mode",
        "lm",
        "--batch_size",
        "1",
        "--microbatch",
        "1",
        "--diffusion_steps",
        "20",
        "--noise_schedule",
        "gaussian_refine_pow2.5",
        "--seq_length",
        "121",
        "--num_channels",
        "32",
        "--in_channel",
        "8",
        "--vocab_size",
        "159",
        "--padding_mode",
        "pad",
        "--lr_anneal_steps",
        "1",
        "--save_interval",
        "1",
        "--log_interval",
        "1",
        "--eval_interval",
        "1000",
        "--dropout",
        "0.0",
        "--use_kl",
        "False",
        "--learn_sigma",
        "False",
        "--predict_xstart",
        "True",
        "--rescale_timesteps",
        "False",
        "--aux_loss",
        "True",
    ]

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        train_cmd,
        cwd=repo_root / "improved-diffusion",
        env=pythonpath_env,
        check=True,
    )

    ema_path = checkpoint_dir / "ema_0.9999_000000.pt"
    assert ema_path.exists()
    assert (checkpoint_dir / "training_args.json").exists()
    assert (checkpoint_dir / "random_emb.torch").exists()
    assert (checkpoint_dir / "vocab.json").exists()

    out_dir = tmp_path / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_cmd = [
        sys.executable,
        "scripts/text_sample.py",
        "--model_path",
        str(ema_path),
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
        "no",
    ]

    subprocess.run(
        sample_cmd,
        cwd=repo_root / "improved-diffusion",
        env=pythonpath_env,
        check=True,
    )

    assert list(out_dir.glob("*.npz"))
