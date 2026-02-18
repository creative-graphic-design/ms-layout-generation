from __future__ import annotations

import os
import random
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

for path in (SRC_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

if "torch._six" not in sys.modules:
    torch_six = types.ModuleType("torch._six")
    torch_six.inf = float("inf")
    sys.modules["torch._six"] = torch_six


def pytest_configure(config):
    random.seed(0)
    try:
        import torch

        torch.manual_seed(0)
    except ImportError:
        pass


@pytest.fixture(scope="session", autouse=True)
def disable_wandb():
    env_overrides = {
        "WANDB_MODE": "disabled",
        "WANDB_SILENT": "true",
    }
    old_env = {key: os.environ.get(key) for key in env_overrides}
    os.environ.update(env_overrides)
    yield
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def layoutformer_data_root() -> Path:
    # LayoutFormer++ datasets for Coarse-to-Fine
    layoutformer_root = REPO_ROOT.parent / "LayoutFormer++"
    datasets_dir = layoutformer_root / "datasets"
    if not datasets_dir.exists():
        pytest.skip(f"LayoutFormer++ datasets not found: {datasets_dir}")
    return datasets_dir


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


@pytest.fixture
def ddp_env():
    env_overrides = {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29501",
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
    }
    old_env = {key: os.environ.get(key) for key in env_overrides}
    os.environ.update(env_overrides)
    yield env_overrides
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


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
