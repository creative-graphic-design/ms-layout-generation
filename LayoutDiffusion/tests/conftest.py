from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ["DIFFUSION_NO_MPI"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
IMPROVED_DIFFUSION_DIR = REPO_ROOT / "improved-diffusion"
TRANSFORMERS_DIR = REPO_ROOT / "transformers" / "src"
EVAL_SRC_DIR = REPO_ROOT / "eval_src"

for path in (
    SRC_DIR,
    IMPROVED_DIFFUSION_DIR,
    TRANSFORMERS_DIR,
    EVAL_SRC_DIR,
    REPO_ROOT,
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def pytest_configure(config):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def layoutdiffusion_data_root(repo_root: Path) -> Path:
    return repo_root / "layoutdiffusion-data"


@pytest.fixture(scope="session")
def pythonpath_env(repo_root: Path) -> dict[str, str]:
    paths = [
        repo_root / "src",
        repo_root / "improved-diffusion",
        repo_root / "transformers" / "src",
        repo_root / "eval_src",
        repo_root,
    ]
    existing = os.environ.get("PYTHONPATH", "")
    combined = os.pathsep.join(
        [str(p) for p in paths] + ([existing] if existing else [])
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = combined
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env["DIFFUSION_NO_MPI"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    return env


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    try:
        import torch as th

        return th.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def gpu_count() -> int:
    try:
        import torch as th

        return th.cuda.device_count() if th.cuda.is_available() else 0
    except ImportError:
        return 0


@pytest.fixture
def gpu_device(gpu_available: bool):
    if not gpu_available:
        pytest.skip("GPU not available")
    import torch as th

    return th.device("cuda:0")


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
    env.pop("DIFFUSION_NO_MPI", None)
    return env


@pytest.fixture(autouse=True)
def seed_everything():
    import random
    import numpy as np

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch as th

        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)
            th.backends.cudnn.deterministic = True
            th.backends.cudnn.benchmark = False
    except ImportError:
        pass

    yield

    try:
        import torch as th

        if th.cuda.is_available():
            th.cuda.empty_cache()
    except ImportError:
        pass


@pytest.fixture(scope="session")
def real_rico_dataset(layoutdiffusion_data_root: Path) -> Path:
    dataset_path = (
        layoutdiffusion_data_root / "data" / "processed_datasets" / "RICO_ltrb_lex"
    )
    if not dataset_path.exists():
        pytest.skip(f"Real dataset not found: {dataset_path}")
    return dataset_path


@pytest.fixture(scope="session")
def real_publaynet_dataset(layoutdiffusion_data_root: Path) -> Path:
    dataset_path = (
        layoutdiffusion_data_root / "data" / "processed_datasets" / "PublayNet_ltrb_lex"
    )
    if not dataset_path.exists():
        pytest.skip(f"Real dataset not found: {dataset_path}")
    return dataset_path


@pytest.fixture(scope="session")
def real_checkpoint_rico(layoutdiffusion_data_root: Path) -> Path:
    checkpoint = (
        layoutdiffusion_data_root
        / "results"
        / "checkpoint"
        / "discrete_gaussian_pow2.5_aux_lex_ltrb_200_fine_4e5"
        / "ema_0.9999_175000.pt"
    )
    if not checkpoint.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint}")
    return checkpoint


@pytest.fixture(scope="session")
def real_checkpoint_pub(layoutdiffusion_data_root: Path) -> Path:
    checkpoint = (
        layoutdiffusion_data_root
        / "results"
        / "checkpoint"
        / "gaussian_refine_pow2.5_aux_lex_ltrb_200_5e5_pub"
        / "ema_0.9999_400000.pt"
    )
    if not checkpoint.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint}")
    return checkpoint


@pytest.fixture
def small_rico_dataset(tmp_path: Path, real_rico_dataset: Path) -> Path:
    """Integration test用: 10サンプル"""
    dest = tmp_path / "small_rico"
    _copy_dataset_subset(real_rico_dataset, dest, train=10, valid=3, test=3)
    return dest


@pytest.fixture
def medium_rico_dataset(tmp_path: Path, real_rico_dataset: Path) -> Path:
    """E2Eテスト用: 1000サンプル"""
    dest = tmp_path / "medium_rico"
    _copy_dataset_subset(real_rico_dataset, dest, train=1000, valid=100, test=100)
    return dest


def _copy_dataset_subset(
    source: Path, dest: Path, *, train: int, valid: int, test: int
) -> None:
    """データセットのサブセットをコピー"""
    dest.mkdir(parents=True, exist_ok=True)
    for split, count in [("train", train), ("valid", valid), ("test", test)]:
        src_file = source / f"src1_{split}.txt"
        if not src_file.exists():
            continue
        lines = src_file.read_text(encoding="utf-8").splitlines()
        subset = lines[:count]
        dest_file = dest / f"src1_{split}.txt"
        dest_file.write_text("\n".join(subset) + "\n", encoding="utf-8")
