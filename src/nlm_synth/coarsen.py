"""Block-mean coarsening, the operation that changes the observation scale."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np

__all__ = ["block_reduce_mean", "multi_scale_coarsen"]


def block_reduce_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Average an array over non-overlapping ``factor x factor`` blocks.

    Rows and columns that do not fill a whole block are trimmed from the
    bottom and right edges. NaN cells are ignored within each block; a block
    that is entirely NaN yields NaN.

    Parameters
    ----------
    arr:
        2-D array.
    factor:
        Block size. ``factor <= 1`` returns a copy.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError("arr must be 2-D")
    factor = int(factor)
    if factor <= 1:
        return arr.copy()

    n_rows, n_cols = arr.shape
    r_fit, c_fit = n_rows - (n_rows % factor), n_cols - (n_cols % factor)
    if r_fit == 0 or c_fit == 0:
        raise ValueError(
            f"factor {factor} exceeds array shape {arr.shape}; no full block fits"
        )

    blocks = arr[:r_fit, :c_fit].reshape(r_fit // factor, factor, c_fit // factor, factor)
    if np.isnan(blocks).any():
        # An all-NaN block legitimately averages to NaN; that is the documented
        # behaviour, so silence NumPy's "Mean of empty slice" warning for it.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            return np.nanmean(blocks, axis=(1, 3))
    return blocks.mean(axis=(1, 3))


def multi_scale_coarsen(
    arr: np.ndarray, factors: Iterable[int]
) -> list[tuple[int, np.ndarray]]:
    """Coarsen an array at several scales.

    Parameters
    ----------
    arr:
        2-D array.
    factors:
        Block sizes. Duplicates are collapsed, values below 1 are dropped, and
        factors too large for the array are skipped rather than raising.

    Returns
    -------
    list of (factor, array)
        Sorted by increasing factor.
    """
    arr = np.asarray(arr, dtype=float)
    max_factor = min(arr.shape) if arr.ndim == 2 else 0

    out: list[tuple[int, np.ndarray]] = []
    for factor in sorted({int(f) for f in factors if int(f) >= 1}):
        if factor > max_factor:
            continue
        out.append((factor, block_reduce_mean(arr, factor)))
    return out
