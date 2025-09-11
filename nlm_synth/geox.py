import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
from rasterio.transform import from_origin

def to_xarray(data: np.ndarray, transform, crs: str, nodata: float = None, name: str = "ndvi"):
    nrow, ncol = data.shape
    rows = np.arange(nrow)
    cols = np.arange(ncol)
    x_centers = np.array([transform * (c + 0.5, 0.5) for c in cols])[:,0]
    y_centers = np.array([transform * (0.5, r + 0.5) for r in rows])[:,1]
    da = xr.DataArray(data, dims=("y", "x"), coords={"y": y_centers, "x": x_centers}, name=name)
    da.rio.write_crs(crs, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    if nodata is not None:
        da.rio.write_nodata(nodata, inplace=True)
    return da

def write_geotiff(path: str, data: np.ndarray, transform, crs: str, nodata: float = None, name: str = "ndvi"):
    da = to_xarray(data, transform, crs, nodata, name=name)
    da.rio.to_raster(path)

def coarsen_xr_mean(da: xr.DataArray, factor: int):
    if factor <= 1:
        return da
    da_c = da.coarsen(y=factor, x=factor, boundary='trim').mean()
    if not da_c.rio.crs and da.rio.crs:
        da_c.rio.write_crs(da.rio.crs, inplace=True)
    return da_c
