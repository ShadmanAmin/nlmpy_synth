"""Spatial-structure generators and quantile mapping onto a target marginal.

The workflow implemented here separates two things that are usually entangled
in a real image:

* **spatial structure** -- produced by a neutral landscape model (NLM) that
  yields a field in ``[0, 1]`` with a controllable degree of autocorrelation;
* **the marginal distribution** -- supplied by the user as a 1-D sample of
  observed values (e.g. NDVI pixels from a real scene).

:func:`synth_ndvi_from_distribution` combines the two: it generates a field,
then rank-maps it onto the empirical distribution of the samples so the output
has (up to ties) exactly the requested marginal while keeping the NLM's spatial
structure.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:  # nlmpy >= 1.1 ships the implementation in a submodule
    from nlmpy import nlmpy as _nlm
except ImportError:  # pragma: no cover - depends on installed nlmpy layout
    import nlmpy as _nlm

__all__ = [
    "perlin_field",
    "random_cluster_binary",
    "rank_map_to_distribution",
    "synth_ndvi_from_distribution",
    "perlin_internal_dim",
    "NEIGHBOURHOODS",
]

#: Neighbourhood structures accepted by :func:`random_cluster_binary`.
NEIGHBOURHOODS = ("4-neighbourhood", "8-neighbourhood", "diagonal")


@contextlib.contextmanager
def _legacy_seed(seed: int | None):
    """Temporarily seed the legacy global NumPy RNG that nlmpy relies on.

    nlmpy draws from ``np.random.*`` directly, so reproducing one of its fields
    means seeding the global generator. Doing that unconditionally would clobber
    the caller's RNG state, so the previous state is saved and restored.
    """
    if seed is None:
        yield
        return
    state = np.random.get_state()
    try:
        np.random.seed(int(seed) % (2**32))
        yield
    finally:
        np.random.set_state(state)


def perlin_internal_dim(periods: Sequence[int], octaves: int, lacunarity: int) -> int:
    """Side length multiple that nlmpy requires internally for these parameters.

    nlmpy generates Perlin noise on a square whose side is a multiple of
    ``lcm(periods[0] * lacunarity**(octaves-1), periods[1] * lacunarity**(octaves-1))``
    and then crops. That multiple grows geometrically with ``octaves``: e.g.
    ``periods=(12, 12), octaves=6, lacunarity=5`` needs a 37500x37500 array
    (11 GB) regardless of the output size you asked for. Use this to reject
    infeasible parameter combinations before allocating.
    """
    exponent = max(int(octaves) - 1, 0)
    row_periods = int(periods[0]) * int(lacunarity) ** exponent
    col_periods = int(periods[1]) * int(lacunarity) ** exponent
    return int(np.lcm(row_periods, col_periods))


def perlin_field(
    nrow: int,
    ncol: int,
    periods: Sequence[int] = (4, 4),
    octaves: int = 4,
    lacunarity: int = 2,
    persistence: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a Perlin-noise field rescaled to ``[0, 1]``.

    Parameters
    ----------
    nrow, ncol:
        Output shape.
    periods:
        Number of periods along (row, column) in the first octave. Must be
        positive integers.
    octaves:
        Number of successively finer noise layers summed together.
    lacunarity:
        Integer factor by which the frequency increases per octave.
    persistence:
        Factor by which the amplitude decays per octave.
    seed:
        Seed for the global NumPy RNG used by nlmpy. ``None`` leaves the RNG
        untouched, in which case results are not reproducible.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(nrow, ncol)`` with values in ``[0, 1]``. A perfectly
        flat field is returned as all ``0.5``.
    """
    if nrow < 1 or ncol < 1:
        raise ValueError("nrow and ncol must be >= 1")
    periods = (int(periods[0]), int(periods[1]))
    if periods[0] < 1 or periods[1] < 1:
        raise ValueError("periods must be positive integers")
    if int(lacunarity) < 1:
        raise ValueError("lacunarity must be a positive integer")

    # Always ask nlmpy for a square and crop here, for two reasons: nlmpy's
    # extractRandomArrayFromSquareArray calls np.random.choice(range(dim - nRow)),
    # which raises "'a' cannot be empty" whenever a non-square request happens to
    # need no padding; and its crop offset is random, so cropping deterministically
    # keeps a given seed tied to a given field.
    side = max(int(nrow), int(ncol))
    with _legacy_seed(seed):
        arr = _nlm.perlinNoise(
            nRow=side,
            nCol=side,
            periods=periods,
            persistence=float(persistence),
            octaves=int(octaves),
            lacunarity=int(lacunarity),
        )

    arr = np.asarray(arr, dtype=float)[:nrow, :ncol]
    a_min, a_max = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max - a_min < 1e-12:
        return np.full(arr.shape, 0.5)
    return (arr - a_min) / (a_max - a_min)


def random_cluster_binary(
    nrow: int,
    ncol: int,
    p: float = 0.5,
    cluster_p: float = 0.58,
    neighbourhood: str = "4-neighbourhood",
    seed: int | None = None,
) -> np.ndarray:
    """Generate a binary random-cluster nearest-neighbour field.

    Parameters
    ----------
    nrow, ncol:
        Output shape.
    p:
        Target proportion of cells set to 1 after thresholding.
    cluster_p:
        nlmpy's own ``p``: the proportion of cells randomly selected to seed
        clusters. Values near the percolation threshold (~0.59 for a
        4-neighbourhood) give the largest, most connected patches.
    neighbourhood:
        Connectivity rule used to grow clusters; one of :data:`NEIGHBOURHOODS`.
    seed:
        Seed for the global NumPy RNG used by nlmpy.

    Notes
    -----
    Earlier versions passed ``cluster_p`` into nlmpy's ``p`` slot *and* dropped
    the requested ``p``, so the output proportion was not controllable. Both
    parameters are now honoured: ``cluster_p`` shapes the clusters, ``p`` sets
    how much of the field ends up as 1.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if not 0.0 < cluster_p <= 1.0:
        raise ValueError("cluster_p must be in (0, 1]")
    if neighbourhood not in NEIGHBOURHOODS:
        raise ValueError(f"neighbourhood must be one of {NEIGHBOURHOODS}")

    with _legacy_seed(seed):
        arr = np.asarray(
            _nlm.randomClusterNN(int(nrow), int(ncol), float(cluster_p), n=neighbourhood),
            dtype=float,
        )

    if p <= 0.0:
        return np.zeros_like(arr)
    if p >= 1.0:
        return np.ones_like(arr)

    # Threshold at the (1 - p) quantile so that ~p of the cells become 1.
    flat = arr.ravel()
    kth = int(np.clip(round((1.0 - p) * flat.size), 0, flat.size - 1))
    thresh = np.partition(flat, kth)[kth]
    return (arr >= thresh).astype(float)


def rank_map_to_distribution(field01: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Quantile-map a field onto the empirical distribution of ``samples``.

    The cell with the *k*-th smallest value in ``field01`` receives the *k*-th
    quantile of ``samples``, so the returned array reproduces the sample
    marginal while preserving the field's spatial ordering.

    Parameters
    ----------
    field01:
        2-D field of arbitrary scale. Only the *ranks* of its values matter.
    samples:
        1-D array of target values. Non-finite entries are dropped.

    Notes
    -----
    This performs true rank mapping. The previous implementation indexed the
    sorted samples by the field's *value* (``idx = value * (n - 1)``), which
    only reproduces the target marginal when the field is already uniform on
    ``[0, 1]``. Perlin fields are min-max rescaled, not rank-uniform -- their
    values are bell-shaped -- so that shortcut over-represented mid-range values
    and truncated both tails of the requested distribution.

    Non-finite cells in ``field01`` are preserved as NaN in the output.
    """
    field01 = np.asarray(field01, dtype=float)
    valid_samples = np.asarray(samples, dtype=float).ravel()
    valid_samples = valid_samples[np.isfinite(valid_samples)]
    if valid_samples.size == 0:
        raise ValueError("samples contains no finite values")

    sorted_vals = np.sort(valid_samples)
    flat = field01.ravel()
    finite = np.isfinite(flat)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return np.full(field01.shape, np.nan)

    # Rank the finite cells, then read off the matching sample quantiles.
    order = np.argsort(flat[finite], kind="stable")
    ranks = np.empty(n_finite, dtype=np.int64)
    if n_finite == 1:
        ranks[order] = sorted_vals.size // 2
    else:
        quantiles = np.arange(n_finite, dtype=float) / (n_finite - 1)
        ranks[order] = np.rint(quantiles * (sorted_vals.size - 1)).astype(np.int64)

    out = np.full(flat.shape, np.nan)
    out[finite] = sorted_vals[ranks]
    return out.reshape(field01.shape)


def synth_ndvi_from_distribution(
    nrow: int,
    ncol: int,
    samples: np.ndarray,
    method: str = "perlin",
    method_kwargs: Mapping[str, Any] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Synthesise a field with a given spatial structure and marginal distribution.

    Parameters
    ----------
    nrow, ncol:
        Output shape.
    samples:
        1-D array defining the target marginal distribution.
    method:
        ``'perlin'`` for pure Perlin noise, or ``'cluster'`` for a random-cluster
        binary field blended with fine-scale Perlin noise. The noise breaks ties
        within patches so the quantile mapping yields a continuous surface
        rather than two discrete levels.
    method_kwargs:
        Parameters forwarded to the generator. For ``'cluster'`` the recognised
        keys are ``p``, ``cluster_p``, ``neighbourhood``, ``cluster_weight`` and
        the Perlin keys ``periods``, ``octaves``, ``lacunarity``, ``persistence``.
    seed:
        Seed for reproducibility.
    """
    kwargs = dict(method_kwargs or {})

    if method == "perlin":
        base = perlin_field(nrow, ncol, seed=seed, **kwargs)
    elif method == "cluster":
        weight = float(kwargs.pop("cluster_weight", 0.7))
        binary = random_cluster_binary(
            nrow,
            ncol,
            p=float(kwargs.pop("p", 0.5)),
            cluster_p=float(kwargs.pop("cluster_p", 0.6)),
            neighbourhood=kwargs.pop("neighbourhood", "4-neighbourhood"),
            seed=seed,
        )
        noise = perlin_field(
            nrow,
            ncol,
            periods=kwargs.pop("periods", (6, 6)),
            octaves=kwargs.pop("octaves", 2),
            lacunarity=kwargs.pop("lacunarity", 2),
            persistence=kwargs.pop("persistence", 0.4),
            seed=None if seed is None else seed + 1,
        )
        if kwargs:
            raise TypeError(f"unexpected method_kwargs for 'cluster': {sorted(kwargs)}")
        base = binary * weight + noise * (1.0 - weight)
    else:
        raise ValueError("Unknown method. Use 'perlin' or 'cluster'.")

    # rank_map_to_distribution uses ranks only, so no rescaling is needed here.
    return rank_map_to_distribution(base, samples)
