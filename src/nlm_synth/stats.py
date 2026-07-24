"""Spatial summary statistics for 2-D fields.

All functions accept NaN-containing arrays and ignore the missing cells.
"""

from __future__ import annotations

import numpy as np

__all__ = ["morans_i", "semivariogram", "semivariogram_1d", "summarize_stats"]


def morans_i(arr2d: np.ndarray) -> float:
    """Global Moran's I under a rook (4-neighbour) contiguity weighting.

    Parameters
    ----------
    arr2d:
        2-D array. NaN cells are excluded, along with any neighbour pair that
        involves one.

    Returns
    -------
    float
        Moran's I, or NaN if the field is constant or has fewer than two
        valid cells.

    Notes
    -----
    Vectorised via array shifts. The previous implementation materialised every
    neighbour pair in a Python list and looped over it, costing roughly
    ``2 * nrow * ncol`` interpreted iterations per call -- about half a million
    for a 512x512 field, and this function runs once per scale per Monte Carlo
    run. The shift-based form below is numerically equivalent and orders of
    magnitude faster.

    Deviations are taken about the mean of the valid cells, as in the standard
    estimator.
    """
    x = np.asarray(arr2d, dtype=float)
    if x.ndim != 2:
        raise ValueError("arr2d must be 2-D")

    valid = np.isfinite(x)
    n = int(valid.sum())
    if n < 2:
        return float("nan")

    z = np.where(valid, x - x[valid].mean(), 0.0)

    # Vertical (row i, i+1) and horizontal (col j, j+1) rook neighbours.
    vert_ok = valid[:-1, :] & valid[1:, :]
    horiz_ok = valid[:, :-1] & valid[:, 1:]
    cross = float(
        np.sum(z[:-1, :] * z[1:, :] * vert_ok) + np.sum(z[:, :-1] * z[:, 1:] * horiz_ok)
    )
    n_pairs = int(vert_ok.sum() + horiz_ok.sum())

    denom = float(np.sum(z * z))
    if denom == 0.0 or n_pairs == 0:
        return float("nan")

    # Weights are symmetric, so each unordered pair contributes twice to both
    # the numerator and the sum of weights; those factors of 2 cancel.
    return (n / n_pairs) * (cross / denom)


def semivariogram(
    arr2d: np.ndarray,
    max_lag: float | None = None,
    step: float = 1.0,
    n_pairs: int = 20_000,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical isotropic semivariogram estimated from random cell pairs.

    Parameters
    ----------
    arr2d:
        2-D array; NaN cells are excluded.
    max_lag:
        Largest separation distance to bin, in cells. Defaults to a quarter of
        the array diagonal.
    step:
        Bin width in cells.
    n_pairs:
        Number of random cell pairs to draw.
    random_state:
        Seed for the sampling RNG.

    Returns
    -------
    (lags, gamma):
        Bin centres and mean semivariance per bin. Empty bins are NaN.
    """
    x = np.asarray(arr2d, dtype=float)
    rng = np.random.default_rng(random_state)

    valid_idx = np.argwhere(np.isfinite(x))
    if valid_idx.shape[0] < 2:
        return np.array([]), np.array([])

    n_valid = valid_idx.shape[0]
    i1 = rng.integers(0, n_valid, size=int(n_pairs))
    i2 = rng.integers(0, n_valid, size=int(n_pairs))
    # Drop self-pairs, which would otherwise pile zero semivariance into lag 0.
    keep = i1 != i2
    i1, i2 = i1[keep], i2[keep]
    if i1.size == 0:
        return np.array([]), np.array([])

    p1, p2 = valid_idx[i1], valid_idx[i2]
    dist = np.hypot(p1[:, 0] - p2[:, 0], p1[:, 1] - p2[:, 1])
    gamma_i = 0.5 * (x[p1[:, 0], p1[:, 1]] - x[p2[:, 0], p2[:, 1]]) ** 2

    if max_lag is None:
        max_lag = float(np.hypot(*x.shape)) / 4.0
    bins = np.arange(0.0, max_lag + step, step)
    if bins.size < 2:
        return np.array([]), np.array([])

    which = np.digitize(dist, bins) - 1
    n_bins = bins.size - 1
    inside = (which >= 0) & (which < n_bins)

    # bincount in a single pass, instead of a per-bin boolean scan of all pairs.
    counts = np.bincount(which[inside], minlength=n_bins).astype(float)
    sums = np.bincount(which[inside], weights=gamma_i[inside], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = np.where(counts > 0, sums / counts, np.nan)

    lags = 0.5 * (bins[:-1] + bins[1:])
    return lags, gamma


#: Backwards-compatible alias for :func:`semivariogram`.
semivariogram_1d = semivariogram


def _semivariogram_range(lags: np.ndarray, gamma: np.ndarray) -> float:
    """Practical range: the first lag reaching 95% of the maximum semivariance."""
    if gamma.size == 0 or not np.any(np.isfinite(gamma)):
        return float("nan")
    sill = np.nanmax(gamma)
    if not np.isfinite(sill) or sill <= 0:
        return float("nan")
    reached = np.flatnonzero(np.nan_to_num(gamma, nan=-np.inf) >= 0.95 * sill)
    return float(lags[reached[0]]) if reached.size else float("nan")


def summarize_stats(
    arr2d: np.ndarray,
    semivar: bool = False,
    **semivar_kwargs,
) -> dict:
    """Summarise a 2-D field into a flat dict of scalar statistics.

    Parameters
    ----------
    arr2d:
        2-D array; NaN cells are excluded from every statistic.
    semivar:
        If True, also fit a semivariogram and report its practical range and
        sill as ``semivar_range`` and ``semivar_sill``. This argument was
        previously accepted and then silently ignored.
    **semivar_kwargs:
        Forwarded to :func:`semivariogram`.

    Returns
    -------
    dict
        Keys ``mean``, ``variance``, ``std_dev``, ``morans_I``, ``n``,
        ``shape_r``, ``shape_c``, plus the two semivariogram keys when
        ``semivar`` is True.
    """
    x = np.asarray(arr2d, dtype=float)
    if x.ndim != 2:
        raise ValueError("arr2d must be 2-D")

    valid = np.isfinite(x)
    n = int(valid.sum())
    if n == 0:
        stats = {
            "mean": float("nan"),
            "variance": float("nan"),
            "std_dev": float("nan"),
            "morans_I": float("nan"),
            "n": 0,
            "shape_r": int(x.shape[0]),
            "shape_c": int(x.shape[1]),
        }
    else:
        values = x[valid]
        stats = {
            "mean": float(values.mean()),
            "variance": float(values.var()),
            "std_dev": float(values.std()),
            "morans_I": float(morans_i(x)),
            "n": n,
            "shape_r": int(x.shape[0]),
            "shape_c": int(x.shape[1]),
        }

    if semivar:
        lags, gamma = semivariogram(x, **semivar_kwargs)
        stats["semivar_range"] = _semivariogram_range(lags, gamma)
        stats["semivar_sill"] = float(np.nanmax(gamma)) if gamma.size else float("nan")

    return stats
