"""Estimate Perlin parameters that reproduce the spatial structure of a real image.

Given an observed raster (e.g. an NDVI scene), a grid search finds the Perlin
parameters whose synthetic fields best match the image's *structure*. Matching
is done on rank-transformed, z-scored fields, so the objective is insensitive to
the image's marginal distribution -- that part is handled separately by
:func:`~nlm_synth.generators.rank_map_to_distribution`.

The objective combines two structure descriptors:

* the radially averaged power spectrum, which captures the mix of spatial
  frequencies present; and
* Moran's I, which pins down short-range autocorrelation.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr

from .generators import perlin_field, perlin_internal_dim
from .stats import morans_i

__all__ = [
    "fit_perlin_parameters_array",
    "fit_perlin_parameters_geotiff",
    "radial_power_spectrum",
    "square_crop_dataarray",
    "DEFAULT_PERIODS_GRID",
    "DEFAULT_OCTAVES_GRID",
    "DEFAULT_LACUNARITY_GRID",
    "DEFAULT_PERSISTENCE_GRID",
    "MAX_INTERNAL_DIM_AUTO",
]

DEFAULT_PERIODS_GRID: tuple[tuple[int, int], ...] = ((2, 2), (3, 3), (4, 4), (6, 6), (8, 8), (12, 12))
DEFAULT_OCTAVES_GRID: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
DEFAULT_LACUNARITY_GRID: tuple[int, ...] = (2, 3, 4)
DEFAULT_PERSISTENCE_GRID: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)

#: Sentinel for ``max_internal_dim``: cap the internal grid at the field's own
#: size. See :func:`_feasible_combos` for why that is the right default.
MAX_INTERNAL_DIM_AUTO = "auto"


# ---------------------------------------------------------------------------
# Structure descriptors
# ---------------------------------------------------------------------------
def _rank01(arr: np.ndarray) -> np.ndarray:
    """Map a 2-D array to ``[0, 1)`` by rank, preserving NaNs."""
    x = np.asarray(arr, dtype=float)
    mask = np.isfinite(x)
    vals = x[mask]
    out = np.full(x.shape, np.nan)
    if vals.size == 0:
        return out
    if vals.size < 2:
        out[mask] = 0.5
        return out
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(vals.size, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, vals.size, endpoint=False)
    out[mask] = ranks
    return out


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance rescaling that ignores NaNs."""
    x = np.asarray(arr, dtype=float)
    mean, std = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(x)
    return (x - mean) / std


def radial_power_spectrum(
    arr: np.ndarray, n_bins: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged, sum-normalised power spectrum of a 2-D field.

    A separable Hann window is applied first to suppress the spectral leakage
    that a non-periodic field would otherwise produce at the array edges. NaNs
    are treated as zero (i.e. as the field mean, once z-scored).

    Returns
    -------
    (freq, power):
        Normalised radial frequency in ``[0, 1]`` and the fraction of total
        power in each annulus. ``power`` sums to 1 over finite bins.

    Notes
    -----
    The previous implementation binned by distance from the array *centre*
    without applying :func:`numpy.fft.fftshift`, but ``fft2`` places the DC
    component at index ``[0, 0]``. Radii were therefore measured from a point
    of the spectrum with no physical meaning, and the resulting "spectra" did
    not distinguish coarse from fine structure. The shift below fixes that.
    """
    x = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0)
    n_rows, n_cols = x.shape

    window = np.hanning(n_rows)[:, None] * np.hanning(n_cols)[None, :]
    power = np.abs(np.fft.fft2(x * window)) ** 2
    # Move the DC component to the array centre so radii are measured from it.
    power = np.fft.fftshift(power)

    center_y, center_x = n_rows // 2, n_cols // 2
    yy, xx = np.indices((n_rows, n_cols))
    radius = np.hypot(yy - center_y, xx - center_x)

    r_max = float(min(center_y, center_x))
    if r_max <= 0:
        return np.array([]), np.array([])

    bins = np.linspace(0.0, r_max, n_bins + 1)
    which = np.digitize(radius.ravel(), bins) - 1
    inside = (which >= 0) & (which < n_bins)

    counts = np.bincount(which[inside], minlength=n_bins).astype(float)
    sums = np.bincount(which[inside], weights=power.ravel()[inside], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        spectrum = np.where(counts > 0, sums / counts, np.nan)

    # Normalise unconditionally so target and candidate spectra are always on
    # the same scale. The old code skipped normalisation whenever any bin was
    # empty, which silently made those comparisons meaningless.
    total = np.nansum(spectrum)
    if total > 0:
        spectrum = spectrum / total

    freq = 0.5 * (bins[:-1] + bins[1:]) / r_max
    return freq, spectrum


def _objective(
    target_ps: np.ndarray,
    target_mi: float,
    cand_ps: np.ndarray,
    cand_mi: float,
    w_spec: float = 1.0,
    w_moran: float = 0.3,
) -> float:
    """Weighted mismatch between two structure descriptors. Lower is better."""
    mask = np.isfinite(target_ps) & np.isfinite(cand_ps)
    spec_err = float(np.mean((target_ps[mask] - cand_ps[mask]) ** 2)) if mask.sum() >= 5 else 1e3
    mi_err = abs(
        (target_mi if np.isfinite(target_mi) else 0.0)
        - (cand_mi if np.isfinite(cand_mi) else 0.0)
    )
    return w_spec * spec_err + w_moran * mi_err


# ---------------------------------------------------------------------------
# Raster preparation
# ---------------------------------------------------------------------------
def square_crop_dataarray(da2d: xr.DataArray, align: str = "center") -> xr.DataArray:
    """Crop a 2-D DataArray to its largest centred (or corner-aligned) square.

    Parameters
    ----------
    da2d:
        DataArray with ``y`` and ``x`` dimensions.
    align:
        One of ``'center'``, ``'ul'``, ``'ur'``, ``'ll'``, ``'lr'``.

    Notes
    -----
    Perlin comparisons are most interpretable on a square domain, since the
    radial power spectrum assumes isotropic axes. Georeferencing is preserved
    because the crop uses ``isel``.
    """
    n_rows, n_cols = int(da2d.sizes["y"]), int(da2d.sizes["x"])
    side = min(n_rows, n_cols)

    offsets = {
        "center": ((n_rows - side) // 2, (n_cols - side) // 2),
        "ul": (0, 0),
        "ur": (0, n_cols - side),
        "ll": (n_rows - side, 0),
        "lr": (n_rows - side, n_cols - side),
    }
    if align not in offsets:
        raise ValueError(f"align must be one of {sorted(offsets)}")
    y0, x0 = offsets[align]
    return da2d.isel(y=slice(y0, y0 + side), x=slice(x0, x0 + side))


def _feasible_combos(
    periods_grid: Iterable[Sequence[int]],
    octaves_grid: Iterable[int],
    lacunarity_grid: Iterable[int],
    persistence_grid: Iterable[float],
    max_internal_dim: int,
) -> tuple[list[tuple], list[tuple]]:
    """Split the parameter grid into usable and out-of-range combinations.

    nlmpy allocates a square whose side is a multiple of
    ``lcm(p_r * L**(o-1), p_c * L**(o-1))``, which explodes for large octaves
    and lacunarity: ``periods=(12, 12), octaves=6, lacunarity=5`` needs a
    37500x37500 array (11 GB) even for a 512x512 output. Screening up front
    turns what used to be an out-of-memory crash mid-search into a reported
    skip.

    Capping the limit at the field's own size (the ``'auto'`` default) is also
    the scientifically right cut-off, not just the cheap one. Once the multiple
    exceeds the field size, nlmpy builds a much larger grid and crops a window
    holding less than one period of the coarsest octave -- a smooth gradient
    whose finest octave aliases at roughly one period per cell. Such candidates
    cost seconds each and describe no structure the field could actually show.
    """
    feasible, skipped = [], []
    for periods, octaves, lacunarity, persistence in itertools.product(
        periods_grid, octaves_grid, lacunarity_grid, persistence_grid
    ):
        combo = ((int(periods[0]), int(periods[1])), int(octaves), int(lacunarity), float(persistence))
        if perlin_internal_dim(combo[0], combo[1], combo[2]) > max_internal_dim:
            skipped.append(combo)
        else:
            feasible.append(combo)
    return feasible, skipped


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_perlin_parameters_array(
    ndvi_arr: np.ndarray,
    periods_grid: Iterable[Sequence[int]] = DEFAULT_PERIODS_GRID,
    octaves_grid: Iterable[int] = DEFAULT_OCTAVES_GRID,
    lacunarity_grid: Iterable[int] = DEFAULT_LACUNARITY_GRID,
    persistence_grid: Iterable[float] = DEFAULT_PERSISTENCE_GRID,
    n_bins_spectrum: int = 60,
    n_repeats: int = 1,
    seed: int = 1234,
    max_internal_dim: int | str = MAX_INTERNAL_DIM_AUTO,
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """Grid-search Perlin parameters matching an array's spatial structure.

    Parameters
    ----------
    ndvi_arr:
        2-D observed field. NaNs are ignored.
    periods_grid, octaves_grid, lacunarity_grid, persistence_grid:
        Candidate values. ``lacunarity`` must be an integer.
    n_bins_spectrum:
        Number of radial frequency bins in the spectral descriptor.
    n_repeats:
        Realisations averaged per candidate. Values above 1 reduce the effect
        of a single unlucky noise draw at proportional cost.
    seed:
        Base seed; realisation ``k`` of a candidate uses ``seed + k``.
    max_internal_dim:
        Skip candidates that would make nlmpy allocate a square larger than
        this. ``'auto'`` (the default) uses the field's own size, which keeps
        every candidate well posed and the search fast; pass an integer to
        explore coarser parameters at the cost of time and memory. See
        :func:`_feasible_combos` and
        :func:`~nlm_synth.generators.perlin_internal_dim`.
    verbose:
        Print progress and each new best candidate as the search proceeds.

    Returns
    -------
    (best, diagnostics):
        ``best`` holds the winning parameters plus ``score``, ``moran`` and
        ``target_moran``. ``diagnostics`` has one dict per evaluated candidate,
        suitable for ``pd.DataFrame(diagnostics)``.

    Raises
    ------
    ValueError
        If the input is not 2-D, is entirely NaN, or if no candidate in the
        grid is feasible under ``max_internal_dim``.
    """
    ndvi = np.asarray(ndvi_arr, dtype=float)
    if ndvi.ndim != 2:
        raise ValueError("ndvi_arr must be 2-D")
    if not np.any(np.isfinite(ndvi)):
        raise ValueError("ndvi_arr contains no finite values")

    n_rows, n_cols = ndvi.shape
    if max_internal_dim == MAX_INTERNAL_DIM_AUTO:
        max_internal_dim = max(n_rows, n_cols)

    # Structure-only target: rank then z-score, so the marginal drops out.
    rank_img = _rank01(ndvi)
    _, target_ps = radial_power_spectrum(_zscore(rank_img), n_bins=n_bins_spectrum)
    target_mi = morans_i(rank_img)
    if verbose:
        print(f"[fit] target Moran's I (rank image): {target_mi:.4f}")

    feasible, skipped = _feasible_combos(
        periods_grid, octaves_grid, lacunarity_grid, persistence_grid, max_internal_dim
    )
    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} of {len(skipped) + len(feasible)} parameter "
            f"combinations needing an internal grid larger than "
            f"max_internal_dim={max_internal_dim}. Such candidates show less than "
            "one period across the field. Raise max_internal_dim to include them, "
            "at the cost of time and memory.",
            stacklevel=2,
        )
    if not feasible:
        raise ValueError(
            "No parameter combination is usable under "
            f"max_internal_dim={max_internal_dim}; reduce octaves/lacunarity "
            "or raise the limit."
        )
    if verbose:
        print(f"[fit] evaluating {len(feasible)} candidate parameter sets")

    best: dict = {"score": np.inf}
    diagnostics: list[dict] = []

    for periods, octaves, lacunarity, persistence in feasible:
        scores, morans = [], []
        for k in range(int(n_repeats)):
            candidate = perlin_field(
                n_rows,
                n_cols,
                periods=periods,
                octaves=octaves,
                lacunarity=lacunarity,
                persistence=persistence,
                seed=seed + k,
            )
            cand_rank = _rank01(candidate)
            _, cand_ps = radial_power_spectrum(_zscore(cand_rank), n_bins=n_bins_spectrum)
            cand_mi = morans_i(cand_rank)
            scores.append(_objective(target_ps, target_mi, cand_ps, cand_mi))
            morans.append(cand_mi)

        score = float(np.mean(scores))
        cand_mi = float(np.nanmean(morans))
        diagnostics.append(
            {
                "periods": periods,
                "octaves": octaves,
                "lacunarity": lacunarity,
                "persistence": persistence,
                "score": score,
                "moran": cand_mi,
            }
        )

        if score < best["score"]:
            best = {
                "periods": periods,
                "octaves": octaves,
                "lacunarity": lacunarity,
                "persistence": persistence,
                "score": score,
                "moran": cand_mi,
                "target_moran": float(target_mi),
            }
            if verbose:
                print(
                    f"[fit] new best: periods={periods} octaves={octaves} "
                    f"lacunarity={lacunarity} persistence={persistence} "
                    f"score={score:.6g} I={cand_mi:.4f}"
                )

    return best, diagnostics


def _diagnostics_frame(diagnostics: list[dict], source: str, band: int) -> pd.DataFrame:
    """Flatten diagnostic rows into a table with scalar columns."""
    return pd.DataFrame(
        [
            {
                "source": source,
                "periods_r": int(d["periods"][0]),
                "periods_c": int(d["periods"][1]),
                "octaves": int(d["octaves"]),
                "lacunarity": int(d["lacunarity"]),
                "persistence": float(d["persistence"]),
                "score": float(d["score"]),
                "candidate_moran": float(d["moran"]),
                "band": band,
            }
            for d in diagnostics
        ]
    )


def fit_perlin_parameters_geotiff(
    in_tif: str,
    out_csv: str | None = None,
    periods_grid: Iterable[Sequence[int]] = DEFAULT_PERIODS_GRID,
    octaves_grid: Iterable[int] = DEFAULT_OCTAVES_GRID,
    lacunarity_grid: Iterable[int] = DEFAULT_LACUNARITY_GRID,
    persistence_grid: Iterable[float] = DEFAULT_PERSISTENCE_GRID,
    n_bins_spectrum: int = 60,
    n_repeats: int = 1,
    seed: int = 1234,
    max_internal_dim: int | str = MAX_INTERNAL_DIM_AUTO,
    verbose: bool = True,
    save_diagnostics_csv: str | None = None,
    band: int = 1,
    square_align: str = "center",
) -> dict:
    """Fit Perlin parameters to a GeoTIFF and optionally write the results.

    The raster is cropped to its largest square, nodata cells are masked to
    NaN, and the fit runs on the result.

    Parameters
    ----------
    in_tif:
        Path to the input raster.
    out_csv:
        If given, write the single best-parameter row here.
    save_diagnostics_csv:
        If given, write the score of every evaluated candidate here.
    band:
        1-based band index for multi-band rasters.
    square_align:
        Which square to keep; see :func:`square_crop_dataarray`.

    Other parameters are passed through to :func:`fit_perlin_parameters_array`.

    Returns
    -------
    dict
        The best-parameter dictionary.
    """
    da = rxr.open_rasterio(in_tif, masked=True)
    try:
        if "band" in da.dims:
            da2d = da.sel(band=band).squeeze()
        else:
            da2d = da.squeeze()

        da2d_sq = square_crop_dataarray(da2d, align=square_align)
        arr = da2d_sq.values.astype(float)
        nodata = da2d_sq.rio.nodata
        if nodata is not None and np.isfinite(nodata):
            arr = np.where(arr == nodata, np.nan, arr)

        if verbose:
            print(
                f"[io] {in_tif}: {tuple(da2d.shape)} -> square crop "
                f"{tuple(da2d_sq.shape)}, CRS={da2d_sq.rio.crs}, nodata={nodata}"
            )
    finally:
        da.close()

    best, diagnostics = fit_perlin_parameters_array(
        arr,
        periods_grid=periods_grid,
        octaves_grid=octaves_grid,
        lacunarity_grid=lacunarity_grid,
        persistence_grid=persistence_grid,
        n_bins_spectrum=n_bins_spectrum,
        n_repeats=n_repeats,
        seed=seed,
        max_internal_dim=max_internal_dim,
        verbose=verbose,
    )

    if out_csv:
        pd.DataFrame(
            [
                {
                    "input_tif": in_tif,
                    "periods_r": int(best["periods"][0]),
                    "periods_c": int(best["periods"][1]),
                    "octaves": int(best["octaves"]),
                    "lacunarity": int(best["lacunarity"]),
                    "persistence": float(best["persistence"]),
                    "score": float(best["score"]),
                    "target_moran": float(best["target_moran"]),
                    "candidate_moran": float(best["moran"]),
                    "n_rows": int(arr.shape[0]),
                    "n_cols": int(arr.shape[1]),
                    "band": int(band),
                }
            ]
        ).to_csv(out_csv, index=False)
        if verbose:
            print(f"[io] wrote best parameters to {out_csv}")

    if save_diagnostics_csv:
        _diagnostics_frame(diagnostics, in_tif, int(band)).to_csv(
            save_diagnostics_csv, index=False
        )
        if verbose:
            print(f"[io] wrote {len(diagnostics)} candidate scores to {save_diagnostics_csv}")

    return best
