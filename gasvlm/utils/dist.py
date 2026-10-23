"""Distributed-evaluation primitives.

Model-agnostic PyTorch DDP helpers plus a disk-based cross-rank result gather
(`collect_results_cpu`), following the MMEngine convention.
"""
from itertools import zip_longest, chain
import os.path as osp
import torch
import os
from torch import distributed as torch_dist
from torch.distributed import ProcessGroup
import functools
from typing import Callable, Optional, Tuple
import pickle
import shutil


def _init_dist_pytorch(backend, **kwargs) -> None:
    """Initialize distributed environment with PyTorch launcher.

    Args:
        backend (str): Backend of torch.distributed. Supported backends are
            'nccl', 'gloo' and 'mpi'. Defaults to 'nccl'.
        **kwargs: keyword arguments are passed to ``init_process_group``.
    """
    # LOCAL_RANK is set by `torch.distributed.launch` since PyTorch 1.1
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    torch_dist.init_process_group(backend=backend, **kwargs)


def get_dist_info(group=None) -> Tuple[int, int]:
    """Get distributed information of the given process group."""
    world_size = get_world_size(group)
    rank = get_rank(group)
    return rank, world_size


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """Return the number of the given process group."""
    if is_distributed():
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """Return the rank of the given process group."""
    if is_distributed():
        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0


def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()


def get_default_group() -> Optional[ProcessGroup]:
    """Return default process group."""
    return torch_dist.distributed_c10d._get_default_group()


def is_main_process(group: Optional[ProcessGroup] = None) -> bool:
    """Whether the current rank of the given process group is equal to 0."""
    return get_rank(group) == 0


def master_only(func: Callable) -> Callable:
    """Decorate methods which should be executed in master process only."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
    return wrapper


def collect_results_cpu(result_part: list,
                        size: int,
                        tmpdir='./dist_test_temp'):
    """Collect results across ranks via a shared filesystem.

    Each rank pickles its local `result_part` to `tmpdir/part_{rank}.pkl`,
    then all ranks hit a barrier; rank 0 reads every shard, interleaves them
    with `zip_longest` (restoring original dataset order), trims to `size`,
    deletes the tmpdir, and returns the merged list. Non-zero ranks return None.

    Args:
        result_part (list): Result list containing this rank's per-sample records.
        size (int): Expected total result count (pre-shard).
        tmpdir (str | None): Shared-filesystem temp dir for gather. Must be
            visible from all ranks.

    Returns:
        list or None: The collected, ordered results on rank 0; None elsewhere.
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir, exist_ok=True)

    with open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb') as f:
        pickle.dump(result_part, f, protocol=2)

    barrier()

    if rank != 0:
        return None
    else:
        part_list = []
        for i in range(world_size):
            path = osp.join(tmpdir, f'part_{i}.pkl')
            if not osp.exists(path):
                raise FileNotFoundError(
                    f'{tmpdir} is not an shared directory for '
                    f'rank {i}, please make sure {tmpdir} is a shared '
                    'directory for all ranks!')
            with open(path, 'rb') as f:
                part_list.append(pickle.load(f))
        zipped_results = zip_longest(*part_list)
        ordered_results = [
            i for i in chain.from_iterable(zipped_results) if i is not None
        ]
        ordered_results = ordered_results[:size]
        shutil.rmtree(tmpdir)
        return ordered_results


def barrier(group: Optional[ProcessGroup] = None) -> None:
    """Synchronize all processes in the given process group."""
    if is_distributed():
        if group is None:
            group = get_default_group()
        torch_dist.barrier(group)
