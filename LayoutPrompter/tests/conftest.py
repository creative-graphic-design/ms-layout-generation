"""Pytest configuration for LayoutPrompter tests"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DATASET_ROOT = REPO_ROOT / "dataset"
DATASET_REPO_URL = "https://huggingface.co/datasets/KyleLin/LayoutPrompter"
DATASET_SUBDIRS = ("posterlayout", "publaynet", "rico", "webui")

for path in (SRC_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def pytest_configure(config):
    """Set random seeds for reproducibility"""
    random.seed(42)
    torch.manual_seed(42)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


def _dataset_ready(root: Path) -> bool:
    return all((root / subdir).exists() for subdir in DATASET_SUBDIRS)


def _clone_dataset(target_root: Path) -> None:
    target_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "lfs", "install"], check=True)
    subprocess.run(
        ["git", "clone", DATASET_REPO_URL, str(target_root)],
        check=True,
    )


def _ensure_dataset(root: Path) -> None:
    if _dataset_ready(root):
        return

    if root.exists() and any(root.iterdir()):
        temp_root = root.parent / f"{root.name}_hf_tmp"
        if temp_root.exists():
            shutil.rmtree(temp_root)
        _clone_dataset(temp_root)
        for subdir in DATASET_SUBDIRS:
            src = temp_root / subdir
            dest = root / subdir
            if not dest.exists() and src.exists():
                shutil.move(str(src), str(dest))
        shutil.rmtree(temp_root)
    else:
        if root.exists() and not any(root.iterdir()):
            shutil.rmtree(root)
        _clone_dataset(root)


@pytest.fixture(scope="session")
def dataset_root(repo_root: Path) -> Path:
    dataset_root = repo_root / "dataset"
    _ensure_dataset(dataset_root)

    alias_root = repo_root / "src" / "dataset"
    if not alias_root.exists():
        alias_root.parent.mkdir(parents=True, exist_ok=True)
        alias_root.symlink_to(dataset_root, target_is_directory=True)
    return dataset_root


@pytest.fixture(scope="session")
def pythonpath_env(repo_root: Path) -> dict[str, str]:
    paths = [repo_root / "src", repo_root]
    existing = os.environ.get("PYTHONPATH", "")
    combined = os.pathsep.join(
        [str(p) for p in paths] + ([existing] if existing else [])
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = combined
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    return env


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    import torch

    available = torch.cuda.is_available()
    if not available:
        raise RuntimeError("GPU is required for this test suite")
    return True


@pytest.fixture(scope="session")
def gpu_count() -> int:
    import torch

    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("GPU is required for this test suite")
    return count


@pytest.fixture
def gpu_device(gpu_available: bool):
    import torch

    return torch.device("cuda:0")


@pytest.fixture
def single_gpu_env(
    pythonpath_env: dict[str, str], gpu_available: bool
) -> dict[str, str]:
    env = pythonpath_env.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


@pytest.fixture
def multi_gpu_env(pythonpath_env: dict[str, str], gpu_count: int) -> dict[str, str]:
    if gpu_count < 2:
        raise RuntimeError(f"Multi-GPU not available (found {gpu_count} GPUs)")
    env = pythonpath_env.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    return env


@pytest.fixture(autouse=True)
def seed_everything():
    import random
    import numpy as np

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    yield

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def sample_dataset():
    """Return sample dataset name"""
    return "rico"


@pytest.fixture
def sample_id2label():
    """Return sample label mapping"""
    return {0: "text", 1: "image", 2: "button", 3: "icon"}


@pytest.fixture
def canvas_size():
    """Return sample canvas size"""
    return (1440, 2560)
