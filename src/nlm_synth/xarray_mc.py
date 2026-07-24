"""Georeferenced variant of the Monte Carlo experiment.

Same design as :mod:`nlm_synth.monte_carlo`, but every realisation is written
out as a GeoTIFF at each scale with the CRS and transform kept consistent, so
the outputs can be fed straight into a GIS or a downstream model.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rasterio.transform import from_origin

from .generators import synth_ndvi_from_distribution
from .geox import coarsen_xr_mean, to_xarray
from .monte_carlo import DEFAULT_COARSEN_FACTORS, _seed_sequence, default_generator_grid
from .stats import summarize_stats

__all__ = ["run_experiments_geotiff", "RESULTS_FILENAME"]

#: Name of the CSV written into ``out_dir``.
RESULTS_FILENAME = "results_mc_geotiff.csv"


def run_experiments_geotiff(
    samples: np.ndarray,
    out_dir: str | os.PathLike,
    nrow: int = 512,
    ncol: int = 512,
    pixel_size: float = 30.0,
    x0: float = 0.0,
    y0: float = 0.0,
    crs: str = "EPSG:32611",
    generator_grid: Sequence[Mapping[str, Any]] | None = None,
    coarsen_factors: Iterable[int] = DEFAULT_COARSEN_FACTORS,
    n_runs: int = 10,
    random_seed: int = 42,
    write_rasters: bool = True,
    name_prefix: str = "ndvi",
    progress: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run the multi-scale experiment and write per-scale GeoTIFFs.

    Parameters
    ----------
    samples:
        1-D array defining the target marginal distribution.
    out_dir:
        Directory for the rasters and the results CSV. Created if absent.
        Rasters are grouped into one subdirectory per generator label.
    nrow, ncol:
        Size of each synthesised field, in cells.
    pixel_size:
        Ground size of a full-resolution cell, in CRS units.
    x0, y0:
        Upper-left corner of the grid, in CRS units.
    crs:
        Coordinate reference system for the outputs.
    generator_grid:
        Sequence of ``{'label', 'method', 'method_kwargs'}`` dicts. Defaults to
        :func:`~nlm_synth.monte_carlo.default_generator_grid`.
    coarsen_factors:
        Block sizes at which to evaluate statistics and write rasters.
    n_runs:
        Realisations per generator.
    random_seed:
        Master seed; per-realisation seeds are derived from it.
    write_rasters:
        Set False to compute statistics without writing any GeoTIFFs, which is
        much faster when only the summary CSV is wanted.
    name_prefix:
        Filename prefix for the rasters.
    progress:
        Print a line per generator as it completes.

    Returns
    -------
    (df, meta):
        Tidy results with one row per (generator, run, factor), and the
        settings needed to reproduce the run. ``df`` is also written to
        ``out_dir / results_mc_geotiff.csv``.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    grid = list(generator_grid) if generator_grid is not None else default_generator_grid()
    if not grid:
        raise ValueError("generator_grid is empty")
    factors = sorted({int(f) for f in coarsen_factors if int(f) >= 1})
    if not factors:
        raise ValueError("coarsen_factors contains no factor >= 1")

    transform = from_origin(x0, y0, pixel_size, pixel_size)
    seeds = _seed_sequence(random_seed, len(grid), n_runs)

    rows: list[dict[str, Any]] = []
    for cfg_idx, cfg in enumerate(grid):
        label = cfg.get("label", f"cfg{cfg_idx}")
        method = cfg["method"]
        kwargs = cfg.get("method_kwargs", {})
        label_dir = out_path / label
        if write_rasters:
            label_dir.mkdir(parents=True, exist_ok=True)

        for run in range(n_runs):
            field = synth_ndvi_from_distribution(
                nrow,
                ncol,
                samples,
                method=method,
                method_kwargs=kwargs,
                seed=int(seeds[cfg_idx, run]),
            )
            da = to_xarray(field, transform, crs, nodata=None, name="ndvi")

            for factor in factors:
                # factor == 1 returns the input unchanged, so the full-resolution
                # raster is written by the same branch as every other scale
                # rather than by a separate special case.
                coarse = coarsen_xr_mean(da, factor=factor)
                if write_rasters:
                    name = f"{name_prefix}_{label}_run{run}_f{factor}.tif"
                    coarse.rio.to_raster(label_dir / name)

                stats = summarize_stats(coarse.values, semivar=False)
                stats.update(
                    run=run,
                    label=label,
                    method=method,
                    factor=int(factor),
                    pixel_size=float(pixel_size * factor),
                    nrow=int(coarse.sizes["y"]),
                    ncol=int(coarse.sizes["x"]),
                )
                rows.append(stats)

        if progress:
            print(f"[mc-geotiff] {label}: {n_runs} runs done ({cfg_idx + 1}/{len(grid)})")

    df = pd.DataFrame(rows)
    df.to_csv(out_path / RESULTS_FILENAME, index=False)

    meta = {
        "generator_grid": grid,
        "coarsen_factors": factors,
        "crs": crs,
        "pixel_size": pixel_size,
        "origin": (x0, y0),
        "nrow": nrow,
        "ncol": ncol,
        "n_runs": n_runs,
        "random_seed": random_seed,
        "out_dir": str(out_path),
    }
    return df, meta
