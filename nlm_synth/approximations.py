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
def _perlin_safe(r, c, *, periods, octaves, lacunarity, persistence, seed):
    """
    Generate a Perlin field of shape (r, c). If NLMPy requires LCM-compatible
    sizes, pad up to the nearest valid size, generate, then crop back.
    """
    # force required dtypes
    pr, pc = int(periods[0]), int(periods[1])
    octaves = int(octaves)
    lacunarity = int(lacunarity)
    persistence = float(persistence)

    # First try the requested size
    try:
        return perlin_field(r, c,
                            periods=(pr, pc),
                            octaves=octaves,
                            lacunarity=lacunarity,
                            persistence=persistence,
                            seed=seed)
    except Exception:
        # Compute a safe pad using LCM of periods (common requirement in NLMPy)
        # Pad each dimension to the next multiple of the LCM.
        try:
            lcm = int(np.lcm(pr, pc))
            r2 = int(np.ceil(r / lcm) * lcm)
            c2 = int(np.ceil(c / lcm) * lcm)
            # Never pad to 0; guarantee at least original size
            r2 = max(r2, r)
            c2 = max(c2, c)
        except Exception:
            # Fallback: pad by one period if LCM goes sideways
            r2 = r + pr
            c2 = c + pc

        # As a last resort, make square if NLMPy build requires it
        # (some builds behave more happily with square dims)
        # Keep this commented unless you still see errors:
        # s2 = max(r2, c2)
        # r2, c2 = s2, s2

        field_big = perlin_field(r2, c2,
                                 periods=(pr, pc),
                                 octaves=octaves,
                                 lacunarity=lacunarity,
                                 persistence=persistence,
                                 seed=seed)
        # Crop back to target window (top-left crop is fine for stats)
        return field_big[:r, :c]
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
):
    """
    Read a GeoTIFF via rioxarray, mask nodata, fit Perlin parameters, and write a CSV with best params.
    If save_diagnostics_csv is given, writes all tested combos with scores.
    """
    da = rxr.open_rasterio(in_tif)
    # pick a single 2D band
    if da.ndim == 3:
        # da dims likely (band, y, x)
        if band is None:
            band = int(da.sizes.get('band', 1) >= 1 and 1) or 1
        if 'band' in da.dims:
            da2d = da.sel(band=band).squeeze()
        else:
            da2d = da[0].squeeze()
    else:
        da2d = da.squeeze()

    nd = da2d.rio.nodata
    arr = da2d.values.astype(float)
    if nd is not None:
        arr = np.where(arr == nd, np.nan, arr)

    if verbose:
        print(f"[io] Loaded {in_tif} -> array shape {arr.shape}, CRS={da2d.rio.crs}, nodata={nd}")

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

    # Write best params CSV
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

    # Optionally write diagnostics grid CSV
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