"""nlm_synth: neutral landscape models with prescribed marginal distributions.

Synthesise rasters that carry a controlled amount of spatial structure while
reproducing an observed value distribution, then measure how their statistics
change as the observation scale coarsens.

The georeferencing modules (:mod:`nlm_synth.geox`, :mod:`nlm_synth.xarray_mc`,
:mod:`nlm_synth.approximations`) need the optional ``geo`` extra, so they are
imported lazily and the core NumPy workflow runs without rasterio installed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .coarsen import block_reduce_mean, multi_scale_coarsen
from .generators import (
    perlin_field,
    perlin_internal_dim,
    random_cluster_binary,
    rank_map_to_distribution,
    synth_ndvi_from_distribution,
)
from .monte_carlo import default_generator_grid, run_experiments
from .stats import morans_i, semivariogram, summarize_stats
from .visualize import plot_field_grid, plot_marginal, plot_metric_by_scale

__version__ = "0.2.0"

#: Names served on demand from optional, geo-dependent submodules.
_LAZY_EXPORTS = {
    "to_xarray": "nlm_synth.geox",
    "write_geotiff": "nlm_synth.geox",
    "coarsen_xr_mean": "nlm_synth.geox",
    "scale_transform": "nlm_synth.geox",
    "run_experiments_geotiff": "nlm_synth.xarray_mc",
    "fit_perlin_parameters_array": "nlm_synth.approximations",
    "fit_perlin_parameters_geotiff": "nlm_synth.approximations",
    "radial_power_spectrum": "nlm_synth.approximations",
}

if TYPE_CHECKING:  # pragma: no cover - re-exported for static analysers
    from .approximations import (  # noqa: F401
        fit_perlin_parameters_array,
        fit_perlin_parameters_geotiff,
        radial_power_spectrum,
    )
    from .geox import coarsen_xr_mean, scale_transform, to_xarray, write_geotiff  # noqa: F401
    from .xarray_mc import run_experiments_geotiff  # noqa: F401


def __getattr__(name: str):
    """Import geo-dependent names on first use, with a clear error if unavailable."""
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"{name!r} lives in {module_path}, which needs the optional geo "
            "dependencies. Install them with `pip install nlm-synth[geo]`."
        ) from exc
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = [
    "__version__",
    # generators
    "perlin_field",
    "random_cluster_binary",
    "rank_map_to_distribution",
    "synth_ndvi_from_distribution",
    "perlin_internal_dim",
    # coarsening
    "block_reduce_mean",
    "multi_scale_coarsen",
    # statistics
    "morans_i",
    "semivariogram",
    "summarize_stats",
    # experiments
    "run_experiments",
    "default_generator_grid",
    # plotting
    "plot_metric_by_scale",
    "plot_field_grid",
    "plot_marginal",
    # optional, geo-dependent
    *_LAZY_EXPORTS,
]
