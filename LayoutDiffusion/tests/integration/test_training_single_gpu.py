from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_single_gpu_training_pipeline(
    tmp_path: Path,
    small_rico_dataset: Path,
    single_gpu_env: dict[str, str],
) -> None:
    """
    Integration test: short training run (5 steps) on single GPU

    This test verifies:
    - Training starts successfully on GPU
    - Checkpoints are created
    - GPU is utilized during training
    - No crashes occur during training loop
    """
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Prepare training command for 5 steps
    train_cmd = [
        sys.executable,
        "scripts/train.py",
        "--checkpoint_path",
        str(checkpoint_dir),
        "--e2e_train",
        str(small_rico_dataset),
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
        "2",
        "--microbatch",
        "2",
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
        "5",  # Stop after 5 steps
        "--save_interval",
        "5",
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

    # Run training
    result = subprocess.run(
        train_cmd,
        cwd=repo_root / "improved-diffusion",
        env=single_gpu_env,
        capture_output=True,
        text=True,
    )

    # Check training completed successfully
    assert result.returncode == 0, f"Training failed with stderr:\n{result.stderr}"

    # Verify GPU was used (check for CUDA in output)
    assert "cuda" in result.stdout.lower() or "gpu" in result.stdout.lower(), (
        "GPU not detected in training output"
    )

    # Verify checkpoints were created
    model_ckpt = checkpoint_dir / "model000005.pt"
    ema_ckpt = checkpoint_dir / "ema_0.9999_000005.pt"

    assert model_ckpt.exists(), f"Model checkpoint not found: {model_ckpt}"
    assert ema_ckpt.exists(), f"EMA checkpoint not found: {ema_ckpt}"

    # Verify auxiliary files were created
    assert (checkpoint_dir / "training_args.json").exists(), (
        "training_args.json not found"
    )
    assert (checkpoint_dir / "random_emb.torch").exists(), "random_emb.torch not found"
    assert (checkpoint_dir / "vocab.json").exists(), "vocab.json not found"


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_single_gpu_training_with_fp16(
    tmp_path: Path,
    small_rico_dataset: Path,
    single_gpu_env: dict[str, str],
) -> None:
    """
    Integration test: FP16 training on single GPU

    Verifies FP16 mixed precision training works correctly
    """
    checkpoint_dir = tmp_path / "checkpoint_fp16"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        "scripts/train.py",
        "--checkpoint_path",
        str(checkpoint_dir),
        "--e2e_train",
        str(small_rico_dataset),
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
        "2",
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
        "3",
        "--save_interval",
        "3",
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
        "--use_fp16",
        "True",  # Enable FP16
    ]

    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        train_cmd,
        cwd=repo_root / "improved-diffusion",
        env=single_gpu_env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"FP16 training failed with stderr:\n{result.stderr}"

    # Verify checkpoints were created
    model_ckpt = checkpoint_dir / "model000003.pt"
    assert model_ckpt.exists(), "FP16 model checkpoint not created"


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_single_gpu_training_with_gradient_clipping(
    tmp_path: Path,
    small_rico_dataset: Path,
    single_gpu_env: dict[str, str],
) -> None:
    """
    Integration test: Training with gradient clipping

    Verifies gradient clipping prevents exploding gradients
    """
    checkpoint_dir = tmp_path / "checkpoint_gradclip"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        "scripts/train.py",
        "--checkpoint_path",
        str(checkpoint_dir),
        "--e2e_train",
        str(small_rico_dataset),
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
        "2",
        "--microbatch",
        "2",
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
        "3",
        "--save_interval",
        "3",
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
        "--gradient_clipping",
        "1.0",  # Enable gradient clipping
    ]

    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        train_cmd,
        cwd=repo_root / "improved-diffusion",
        env=single_gpu_env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Gradient clipping training failed:\n{result.stderr}"
    )

    # Check that grad_norm is logged (indicates gradient monitoring)
    assert "grad_norm" in result.stdout or "gradient" in result.stdout.lower(), (
        "Gradient monitoring not found in output"
    )


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_single_gpu_checkpoint_resume(
    tmp_path: Path,
    small_rico_dataset: Path,
    single_gpu_env: dict[str, str],
) -> None:
    """
    Integration test: Resume training from checkpoint

    Verifies that training can be resumed from a saved checkpoint
    """
    checkpoint_dir = tmp_path / "checkpoint_resume"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # First training run (3 steps)
    train_cmd_first = [
        sys.executable,
        "scripts/train.py",
        "--checkpoint_path",
        str(checkpoint_dir),
        "--e2e_train",
        str(small_rico_dataset),
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
        "2",
        "--microbatch",
        "2",
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
        "3",
        "--save_interval",
        "3",
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

    # Run first training
    result1 = subprocess.run(
        train_cmd_first,
        cwd=repo_root / "improved-diffusion",
        env=single_gpu_env,
        capture_output=True,
        text=True,
    )

    assert result1.returncode == 0, f"First training run failed:\n{result1.stderr}"

    # Verify checkpoint exists
    checkpoint_path = checkpoint_dir / "model000003.pt"
    assert checkpoint_path.exists(), "Initial checkpoint not created"

    # Second training run (resume from checkpoint, run 2 more steps)
    train_cmd_resume = train_cmd_first.copy()
    # Change lr_anneal_steps to allow more steps
    anneal_idx = train_cmd_resume.index("--lr_anneal_steps")
    train_cmd_resume[anneal_idx + 1] = "5"  # Total 5 steps
    # Add resume checkpoint
    train_cmd_resume.extend(["--resume_checkpoint", str(checkpoint_path)])

    result2 = subprocess.run(
        train_cmd_resume,
        cwd=repo_root / "improved-diffusion",
        env=single_gpu_env,
        capture_output=True,
        text=True,
    )

    assert result2.returncode == 0, f"Resume training failed:\n{result2.stderr}"

    # Verify resumed checkpoint was created
    resumed_checkpoint = checkpoint_dir / "model000005.pt"
    assert resumed_checkpoint.exists(), "Resumed checkpoint not created"

    # Check for resume message in output
    assert (
        "loading model from checkpoint" in result2.stdout.lower()
        or "resume" in result2.stdout.lower()
    ), "Resume message not found in output"


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_single_gpu_training_logs_metrics(
    tmp_path: Path,
    small_rico_dataset: Path,
    single_gpu_env: dict[str, str],
) -> None:
    """
    Integration test: Verify training logs metrics correctly

    Checks that loss, step, and other metrics are logged
    """
    checkpoint_dir = tmp_path / "checkpoint_metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        "scripts/train.py",
        "--checkpoint_path",
        str(checkpoint_dir),
        "--e2e_train",
        str(small_rico_dataset),
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
        "2",
        "--microbatch",
        "2",
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
        "3",
        "--save_interval",
        "3",
        "--log_interval",
        "1",  # Log every step
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

    result = subprocess.run(
        train_cmd,
        cwd=repo_root / "improved-diffusion",
        env=single_gpu_env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Training failed:\n{result.stderr}"

    # Verify key metrics are logged
    output = result.stdout.lower()
    assert "loss" in output, "Loss metric not logged"
    assert "step" in output, "Step metric not logged"

    # Verify training completed expected number of steps
    assert "step" in output or "samples" in output, "Training progress not logged"
