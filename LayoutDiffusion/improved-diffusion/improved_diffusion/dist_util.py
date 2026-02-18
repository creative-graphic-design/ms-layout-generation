"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


class _DummyComm:
    rank = 0
    size = 1

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, value, root=0):
        return value


class _DummyMPI:
    _IS_DUMMY = True
    COMM_WORLD = _DummyComm()


def _get_mpi():
    if os.environ.get("DIFFUSION_NO_MPI", "").lower() in {"1", "true", "yes"}:
        return _DummyMPI
    try:
        from mpi4py import MPI
    except Exception:
        return _DummyMPI
    return MPI


def _using_dummy_mpi(mpi) -> bool:
    return bool(getattr(mpi, "_IS_DUMMY", False))


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    if not dist.is_available():
        return

    mpi = _get_mpi()
    comm = mpi.COMM_WORLD
    backend = "gloo" if _using_dummy_mpi(mpi) or not th.cuda.is_available() else "nccl"

    if _using_dummy_mpi(mpi):
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend=backend, init_method="env://")
        return

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        mpi = _get_mpi()
        return th.device(f"cuda:{mpi.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    mpi = _get_mpi()
    if mpi.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = mpi.COMM_WORLD.bcast(data)
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p.data, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
