import numpy as np
import pandas as pd
import warnings
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt

from nlm_synth.generators import perlin_field
from nlm_synth.stats import morans_i

# ============================================================
# Utilities
# ============================================================
def _rank01(arr):
    """Map a 2D array to [0,1] by rank (ignores NaNs)."""
    x = arr.astype(float).copy()
    mask = ~np.isnan(x)
    vals = x[mask]
    if vals.size < 2:
        y = np.zeros_like(x)
        y[mask] = 0.5
        return y
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0, 1, len(vals), endpoint=False)
    out = np.full_like(x, np.nan, dtype=float)
    out[mask] = ranks
    return out

def _zscore(arr):
    """Zero-mean, unit-variance (ignores NaNs)."""
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(arr)
    return (arr - m) / s

def _radial_power_spectrum(arr, n_bins=60):
    """Radially averaged, normalized power spectrum for a 2D field (NaNs->0)."""
    x = np.nan_to_num(arr, nan=0.0)
    wy = np.hanning(x.shape[0])[:, None]
    wx = np.hanning(x.shape[1])[None, :]
    w  = wy * wx
    X  = np.fft.fft2(x * w)
    P  = np.abs(X)**2

    r, c = arr.shape
    cy, cx = (r//2, c//2)
    yy, xx = np.indices((r, c))
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    rmax = min(cy, cx)
    bins = np.linspace(0, rmax, n_bins+1)
    which = np.digitize(rr, bins) - 1
    ps = np.array([P[which == k].mean() if np.any(which == k) else np.nan for k in range(n_bins)])
    freq = 0.5 * (bins[:-1] + bins[1:]) / rmax  # normalized 0..1

    if np.all(~np.isnan(ps)):
        total = np.nansum(ps)
        if total > 0:
            ps = ps / total
    return freq, ps

def _periods_token(p):
    return (int(p[0]), int(p[1]))

def _objective(target_ps, target_mi, cand_ps, cand_mi, w_spec=1.0, w_moran=0.3):
    """Lower is better."""
    mask = np.isfinite(target_ps) & np.isfinite(cand_ps)
    if mask.sum() < 5:
        spec_err = 1e3
    else:
        spec_err = np.mean((target_ps[mask] - cand_ps[mask])**2)
    mi_err = np.abs((target_mi if np.isfinite(target_mi) else 0.0) -
                    (cand_mi   if np.isfinite(cand_mi)   else 0.0))
    return w_spec * spec_err + w_moran * mi_err
# def _perlin_safe(r, c, *, periods, octaves, lacunarity, persistence, seed):
#     """
#     Generate a Perlin field of shape (r, c). If NLMPy requires LCM-compatible
#     sizes, pad up to the nearest valid size, generate, then crop back.
#     """
#     # force required dtypes
#     pr, pc = int(periods[0]), int(periods[1])
#     octaves = int(octaves)
#     lacunarity = int(lacunarity)
#     persistence = float(persistence)

#     # First try the requested size
#     try:
#         return perlin_field(r, c,
#                             periods=(pr, pc),
#                             octaves=octaves,
#                             lacunarity=lacunarity,
#                             persistence=persistence,
#                             seed=seed)
#     except Exception:
#         # Compute a safe pad using LCM of periods (common requirement in NLMPy)
#         # Pad each dimension to the next multiple of the LCM.
#         try:
#             lcm = int(np.lcm(pr, pc))
#             r2 = int(np.ceil(r / lcm) * lcm)
#             c2 = int(np.ceil(c / lcm) * lcm)
#             # Never pad to 0; guarantee at least original size
#             r2 = max(r2, r)
#             c2 = max(c2, c)
#         except Exception:
#             # Fallback: pad by one period if LCM goes sideways
#             r2 = r + pr
#             c2 = c + pc

#         # As a last resort, make square if NLMPy build requires it
#         # (some builds behave more happily with square dims)
#         # Keep this commented unless you still see errors:
#         # s2 = max(r2, c2)
#         # r2, c2 = s2, s2

#         field_big = perlin_field(r2, c2,
#                                  periods=(pr, pc),
#                                  octaves=octaves,
#                                  lacunarity=lacunarity,
#                                  persistence=persistence,
#                                  seed=seed)
#         # Crop back to target window (top-left crop is fine for stats)
#         return field_big[:r, :c]
def _largest_square_crop_slices(n_rows: int, n_cols: int, align: str = "center"):
    """
    Return (yslice, xslice) that crops an array of shape (n_rows, n_cols)
    to the largest possible square. align ∈ {"center","ul","ur","ll","lr"}.
    """
    side = min(n_rows, n_cols)
    if align == "center":
        y0 = (n_rows - side) // 2
        x0 = (n_cols - side) // 2
    elif align == "ul":  # upper-left
        y0, x0 = 0, 0
    elif align == "ur":  # upper-right
        y0, x0 = 0, n_cols - side
    elif align == "ll":  # lower-left
        y0, x0 = n_rows - side, 0
    elif align == "lr":  # lower-right
        y0, x0 = n_rows - side, n_cols - side
    else:
        raise ValueError("align must be one of {'center','ul','ur','ll','lr'}")
    return slice(y0, y0 + side), slice(x0, x0 + side)


def _max_periods_multiple(periods_grid, octaves_grid, lacunarity_grid):
    """
    Compute the maximum periodsMultiple = lcm(pr * L^(o-1), pc * L^(o-1))
    across the provided grid. This gives a safe modulus to align the square size.
    """
    max_mult = 1
    for p in periods_grid:
        pr, pc = int(p[0]), int(p[1])
        for o in octaves_grid:
            o = int(o)
            for L in lacunarity_grid:
                L = int(L)
                rP = pr * (L ** max(o-1, 0))
                cP = pc * (L ** max(o-1, 0))
                mult = int(np.lcm(rP, cP))
                max_mult = int(np.lcm(max_mult, mult)) if mult > 0 else max_mult
    # guard
    return max(1, max_mult)


def square_crop_dataarray(
    da2d: xr.DataArray,
    *,
    align: str = "center",
    lcm_align: bool = True,
    periods_grid=((2,2),(3,3),(4,4),(6,6),(8,8),(12,12)),
    octaves_grid=(1,2,3,4,5,6),
    lacunarity_grid=(2,3,4,5),
):
    """
    Crop a 2D DataArray to the largest square (optionally also to a side length
    that is a multiple of the max periodsMultiple implied by the grid).
    Preserves CRS/transform.
    """
    n_rows = da2d.sizes["y"]
    n_cols = da2d.sizes["x"]

    # 1) largest square
    side = min(n_rows, n_cols)

    # 2) optional LCM alignment (shrinks the side to nearest lower multiple)
    if lcm_align:
        mult = _max_periods_multiple(periods_grid, octaves_grid, lacunarity_grid)
        if side >= mult:
            side = (side // mult) * mult  # floor to multiple
            if side == 0:  # extremely small images
                side = min(n_rows, n_cols)

    yslice, xslice = _largest_square_crop_slices(n_rows, n_cols, align=align)
    # refit slices to new 'side' after optional LCM alignment
    if align == "center":
        y0 = (n_rows - side) // 2
        x0 = (n_cols - side) // 2
        yslice, xslice = slice(y0, y0 + side), slice(x0, x0 + side)
    else:
        # keep the chosen corner; just ensure side length
        if align == "ul":
            yslice, xslice = slice(0, side), slice(0, side)
        elif align == "ur":
            yslice, xslice = slice(0, side), slice(n_cols - side, n_cols)
        elif align == "ll":
            yslice, xslice = slice(n_rows - side, n_rows), slice(0, side)
        elif align == "lr":
            yslice, xslice = slice(n_rows - side, n_rows), slice(n_cols - side, n_cols)

    # Use isel so rioxarray preserves georeferencing properly
    da_sq = da2d.isel(y=yslice, x=xslice)
    return da_sq
# ============================================================
# Core fitting (array in, params out)
# ============================================================
def fit_perlin_parameters_array(
    ndvi_arr,
    periods_grid=((2,2),(3,3),(4,4),(6,6),(8,8),(12,12)),
    octaves_grid=(1,2,3,4,5,6),
    lacunarity_grid=(2,3,4,5),      # integers (your env)
    persistence_grid=(0.3,0.5,0.7,0.9),
    n_bins_spectrum=60,
    seed=1234,
    verbose=True,
):
    """
    Estimate Perlin parameters by matching isotropic power spectrum + Moran's I (on rank image).
    Returns: dict(best_params), list(diagnostics rows)
    """
    ndvi = np.array(ndvi_arr, dtype=float)
    if ndvi.ndim != 2:
        raise ValueError("ndvi_arr must be 2D")
    if np.all(np.isnan(ndvi)):
        raise ValueError("ndvi_arr contains only NaNs")

    # structure-only target
    rank_img = _rank01(ndvi)
    tgt_field = _zscore(rank_img)

    tgt_freq, tgt_ps = _radial_power_spectrum(tgt_field, n_bins=n_bins_spectrum)
    tgt_mi = morans_i(rank_img)
    if verbose:
        print(f"[fit] Target Moran's I (rank): {tgt_mi:.4f}")

    best = dict(score=np.inf)
    diagnostics = []
    r, c = ndvi.shape

    for p in periods_grid:
        p = _periods_token(p)
        for o in octaves_grid:
            o = int(o)
            for L in lacunarity_grid:
                L = int(L)
                for P in persistence_grid:
                    P = float(P)

                    fld = perlin_field(r, c, periods=p, octaves=o, lacunarity=L, persistence=P, seed=seed)

                    fld_rank = _rank01(fld)
                    fld_field = _zscore(fld_rank)

                    _, cand_ps = _radial_power_spectrum(fld_field, n_bins=n_bins_spectrum)
                    cand_mi = morans_i(fld_rank)

                    score = _objective(tgt_ps, tgt_mi, cand_ps, cand_mi,
                                       w_spec=1.0, w_moran=0.3)

                    row = {
                        'periods': p, 'octaves': o, 'lacunarity': L, 'persistence': P,
                        'score': score, 'moran': cand_mi
                    }
                    diagnostics.append(row)

                    if score < best['score']:
                        best = {
                            'periods': p, 'octaves': o, 'lacunarity': L, 'persistence': P,
                            'score': score, 'moran': cand_mi, 'target_moran': tgt_mi
                        }
                        if verbose:
                            print(f"[fit] New best: p={p}, o={o}, L={L}, P={P}, "
                                  f"score={score:.6f}, MI={cand_mi:.4f}")

    return best, diagnostics

# ============================================================
# GeoTIFF/xarray wrapper
# ============================================================
def fit_perlin_parameters_geotiff(
    in_tif: str,
    out_csv: str,
    periods_grid=((2,2),(3,3),(4,4),(6,6),(8,8),(12,12)),
    octaves_grid=(1,2,3,4,5,6),
    lacunarity_grid=(2,3,4,5),       # ints
    persistence_grid=(0.3,0.5,0.7,0.9),
    n_bins_spectrum=60,
    seed=1234,
    verbose=True,
    save_diagnostics_csv: str | None = None,
    band: int | None = None,
    square_align: str = "center",     # "center","ul","ur","ll","lr"
    lcm_align: bool = True,           # align square side to max periodsMultiple
):
    da = rxr.open_rasterio(in_tif)

    # pick a single 2D band
    if da.ndim == 3:
        if band is None:
            band = 1
        if 'band' in da.dims:
            da2d = da.sel(band=band).squeeze()
        else:
            da2d = da[band-1].squeeze()
    else:
        da2d = da.squeeze()

    # ---- NEW: crop to largest square (optionally aligned to grid LCM) ----
    da2d_sq = square_crop_dataarray(
        da2d,
        align=square_align,
        lcm_align=lcm_align,
        periods_grid=periods_grid,
        octaves_grid=octaves_grid,
        lacunarity_grid=lacunarity_grid,
    )

    # nodata mask AFTER crop
    nd = da2d_sq.rio.nodata
    arr = da2d_sq.values.astype(float)
    if nd is not None:
        arr = np.where(arr == nd, np.nan, arr)

    if verbose:
        print(f"[io] Loaded {in_tif} -> original {tuple(da2d.shape)}, "
              f"square-crop {tuple(da2d_sq.shape)}, CRS={da2d_sq.rio.crs}, nodata={nd}")

    best, diagnostics = fit_perlin_parameters_array(
        arr,
        periods_grid=periods_grid,
        octaves_grid=octaves_grid,
        lacunarity_grid=lacunarity_grid,
        persistence_grid=persistence_grid,
        n_bins_spectrum=n_bins_spectrum,
        seed=seed,
        verbose=verbose,
    )

    # Write best params CSV (unchanged)
    best_row = {
        'input_tif': in_tif,
        'periods_r': int(best['periods'][0]),
        'periods_c': int(best['periods'][1]),
        'octaves': int(best['octaves']),
        'lacunarity': int(best['lacunarity']),
        'persistence': float(best['persistence']),
        'score': float(best['score']),
        'target_moran': float(best['target_moran']),
        'candidate_moran': float(best['moran']),
        'n_rows': int(arr.shape[0]),
        'n_cols': int(arr.shape[1]),
        'band': band if band is not None else 1
    }
    pd.DataFrame([best_row]).to_csv(out_csv, index=False)
    if verbose:
        print(f"[io] Wrote best parameters to {out_csv}")

    if save_diagnostics_csv:
        diag_rows = []
        for d in diagnostics:
            diag_rows.append({
                'input_tif': in_tif,
                'periods_r': int(d['periods'][0]),
                'periods_c': int(d['periods'][1]),
                'octaves': int(d['octaves']),
                'lacunarity': int(d['lacunarity']),
                'persistence': float(d['persistence']),
                'score': float(d['score']),
                'candidate_moran': float(d['moran']),
                'band': band if band is not None else 1
            })
        pd.DataFrame(diag_rows).to_csv(save_diagnostics_csv, index=False)
        if verbose:
            print(f"[io] Wrote diagnostics grid to {save_diagnostics_csv}")

    return best