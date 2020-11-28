r"""Randomness utilites."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    r"""Do best effort to ensure reproducibility on same machine.

    Set random seed on :py:mod:`random` module, :py:mod:`numpy.random`,
    :py:func:`torch.manual_seed` and :py:mod:`torch.cuda`.

    Parameters
    ==========
    seed: int
        Control random seed and let experiment reproducible. Must be bigger
        than or equal to `1`.

    Raises
    ======
    TypeError
        When `seed` is not an instance of `int`.

    Notes
    =====
    Reproducibility is not guaranteed accross different python/numpy/pytorch
    release, different os platforms or different hardwares (including CPUs and
    GPUs).
    """
    # Type check.
    if not isinstance(seed, int):
        raise TypeError('`seed` must be an instance of `int`.')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Disable cuDNN benchmark for deterministic selection on algorithm.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
