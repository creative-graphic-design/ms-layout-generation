from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

for path in (SRC_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_root(repo_root: Path) -> Path:
    return repo_root / "LayoutFormer" / "datasets"


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
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        return 0


@pytest.fixture
def gpu_device(gpu_available: bool):
    if not gpu_available:
        pytest.skip("GPU not available")
    import torch

    return torch.device("cuda:0")


@pytest.fixture
def single_gpu_env(
    pythonpath_env: dict[str, str], gpu_available: bool
) -> dict[str, str]:
    if not gpu_available:
        pytest.skip("GPU not available")
    env = pythonpath_env.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


@pytest.fixture
def multi_gpu_env(pythonpath_env: dict[str, str], gpu_count: int) -> dict[str, str]:
    if gpu_count < 2:
        pytest.skip(f"Multi-GPU not available (found {gpu_count} GPUs)")
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

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    yield

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _find_sample(data, min_labels: int = 2):
    for item in data:
        if len(item["labels"]) >= min_labels:
            return item
    return data[0]


@pytest.fixture(scope="session")
def rico_sample(data_root: Path):
    data = torch.load(data_root / "rico" / "pre_processed_20_25" / "val.pt")
    return _find_sample(data)


@pytest.fixture(scope="session")
def publaynet_sample(data_root: Path):
    data = torch.load(data_root / "publaynet" / "pre_processed_20_5" / "val.pt")
    return _find_sample(data)
