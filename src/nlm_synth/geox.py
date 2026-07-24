"""Georeferencing helpers: NumPy arrays to CRS-aware xarray / GeoTIFF."""

from __future__ import annotations

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
import xarray as xr
from affine import Affine

__all__ = ["to_xarray", "write_geotiff", "coarsen_xr_mean", "scale_transform"]


def to_xarray(
    data: np.ndarray,
    transform: Affine,
    crs: str,
    nodata: float | None = None,
    name: str = "ndvi",
) -> xr.DataArray:
    """Wrap a 2-D array as a georeferenced :class:`xarray.DataArray`.

    Coordinates are cell centres derived from the affine ``transform``.

    Parameters
    ----------
    data:
        2-D array.
    transform:
        Affine geotransform mapping (column, row) to (x, y).
    crs:
        Coordinate reference system, e.g. ``"EPSG:32611"``.
    nodata:
        Optional nodata value recorded in the raster metadata.
    name:
        Name given to the DataArray.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be 2-D")
    n_rows, n_cols = data.shape

    # Cell centres in one vectorised pass. This previously evaluated the affine
    # transform once per row and once per column inside a Python list
    # comprehension, which dominated setup cost for large grids.
    cols = np.arange(n_cols, dtype=float) + 0.5
    rows = np.arange(n_rows, dtype=float) + 0.5
    x_centers = transform.c + cols * transform.a + 0.5 * transform.b
    y_centers = transform.f + 0.5 * transform.d + rows * transform.e

    da = xr.DataArray(
        data, dims=("y", "x"), coords={"y": y_centers, "x": x_centers}, name=name
    )
    da.rio.write_crs(crs, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    if nodata is not None:
        da.rio.write_nodata(nodata, inplace=True)
    return da


def write_geotiff(
    path: str,
    data: np.ndarray,
    transform: Affine,
    crs: str,
    nodata: float | None = None,
    name: str = "ndvi",
) -> None:
    """Write a 2-D array to a GeoTIFF with the given transform and CRS."""
    to_xarray(data, transform, crs, nodata, name=name).rio.to_raster(path)


def scale_transform(transform: Affine, factor: int) -> Affine:
    """Return ``transform`` with pixel size multiplied by ``factor``.

    The origin is unchanged, matching what block-mean coarsening does to the
    grid: the upper-left corner stays put and each cell covers ``factor`` times
    more ground in each direction.
    """
    return transform * Affine.scale(float(factor), float(factor))


def coarsen_xr_mean(da: xr.DataArray, factor: int) -> xr.DataArray:
    """Block-mean a georeferenced DataArray, updating CRS and transform.

    Partial blocks at the bottom/right edges are trimmed. ``factor <= 1``
    returns the input unchanged.
    """
    factor = int(factor)
    if factor <= 1:
        return da

    coarse = da.coarsen(y=factor, x=factor, boundary="trim").mean()
    if da.rio.crs is not None:
        coarse.rio.write_crs(da.rio.crs, inplace=True)
    coarse.rio.write_transform(scale_transform(da.rio.transform(), factor), inplace=True)
    return coarse
