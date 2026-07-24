"""Monte Carlo experiments: how do landscape statistics change with pixel size?

Each experiment repeatedly synthesises a field with a known spatial structure
and a fixed marginal distribution, coarsens it to a range of pixel sizes, and
records summary statistics at every scale. Comparing the resulting curves
across generators isolates the effect of spatial structure on scale-dependent
behaviour.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .coarsen import multi_scale_coarsen
from .generators import synth_ndvi_from_distribution
from .stats import summarize_stats

__all__ = ["run_experiments", "default_generator_grid", "DEFAULT_COARSEN_FACTORS"]

#: Coarsening factors used when none are supplied.
DEFAULT_COARSEN_FACTORS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


def default_generator_grid() -> list[dict[str, Any]]:
    """Four reference generators spanning low to high spatial frequency.

    Returned fresh on each call so callers can mutate the result safely.
    """
    return [
        {
            "label": "perlin_LF",
            "method": "perlin",
            "method_kwargs": dict(periods=(2, 2), octaves=3, lacunarity=2, persistence=0.7),
        },
        {
            "label": "perlin_MF",
            "method": "perlin",
            "method_kwargs": dict(periods=(4, 4), octaves=5, lacunarity=2, persistence=0.6),
        },
        {
            "label": "perlin_HF",
            "method": "perlin",
            "method_kwargs": dict(periods=(8, 8), octaves=6, lacunarity=2, persistence=0.5),
        },
        {
            "label": "cluster_nn",
            "method": "cluster",
            "method_kwargs": dict(
                p=0.55,
                cluster_p=0.65,
                periods=(6, 6),
                octaves=2,
                lacunarity=2,
                persistence=0.4,
            ),
        },
    ]


def _seed_sequence(random_seed: int, n_configs: int, n_runs: int) -> np.ndarray:
    """Per-(config, run) seeds drawn once, so runs are independent of iteration order.

    Drawing all seeds up front means adding a generator to the grid does not
    shift the seeds used by the others.
    """
    rng = np.random.default_rng(random_seed)
    return rng.integers(0, 2**31 - 1, size=(n_configs, n_runs), dtype=np.int64)


def run_experiments(
    samples: np.ndarray,
    nrow: int = 512,
    ncol: int = 512,
    generator_grid: Sequence[Mapping[str, Any]] | None = None,
    coarsen_factors: Iterable[int] = DEFAULT_COARSEN_FACTORS,
    n_runs: int = 30,
    semivar: bool = False,
    random_seed: int = 42,
    progress: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run the NumPy-only multi-scale Monte Carlo experiment.

    Parameters
    ----------
    samples:
        1-D array defining the target marginal distribution.
    nrow, ncol:
        Size of each synthesised field.
    generator_grid:
        Sequence of ``{'label', 'method', 'method_kwargs'}`` dicts. Defaults to
        :func:`default_generator_grid`.
    coarsen_factors:
        Block sizes at which to evaluate statistics.
    n_runs:
        Realisations per generator.
    semivar:
        Also compute semivariogram range and sill at each scale.
    random_seed:
        Master seed; per-realisation seeds are derived from it.
    progress:
        Print a line per generator as it completes.

    Returns
    -------
    (df, meta):
        ``df`` is tidy -- one row per (generator, run, coarsening factor) --
        with the columns from :func:`~nlm_synth.stats.summarize_stats` plus
        ``run``, ``label``, ``method``, ``factor`` and ``pixel_size_rel``.
        ``meta`` records the settings needed to reproduce the run.
    """
    grid = list(generator_grid) if generator_grid is not None else default_generator_grid()
    if not grid:
        raise ValueError("generator_grid is empty")
    factors = list(coarsen_factors)
    seeds = _seed_sequence(random_seed, len(grid), n_runs)

    rows: list[dict[str, Any]] = []
    for cfg_idx, cfg in enumerate(grid):
        label = cfg.get("label", f"cfg{cfg_idx}")
        method = cfg["method"]
        kwargs = cfg.get("method_kwargs", {})

        for run in range(n_runs):
            field = synth_ndvi_from_distribution(
                nrow,
                ncol,
                samples,
                method=method,
                method_kwargs=kwargs,
                seed=int(seeds[cfg_idx, run]),
            )
            for factor, coarse in multi_scale_coarsen(field, factors):
                stats = summarize_stats(coarse, semivar=semivar)
                stats.update(
                    run=run,
                    label=label,
                    method=method,
                    factor=int(factor),
                    pixel_size_rel=float(factor),
                )
                rows.append(stats)

        if progress:
            print(f"[mc] {label}: {n_runs} runs done ({cfg_idx + 1}/{len(grid)})")

    meta = {
        "generator_grid": grid,
        "coarsen_factors": factors,
        "nrow": nrow,
        "ncol": ncol,
        "n_runs": n_runs,
        "random_seed": random_seed,
        "semivar": semivar,
    }
    return pd.DataFrame(rows), meta
