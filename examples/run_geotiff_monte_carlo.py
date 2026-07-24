"""Multi-scale Monte Carlo experiment writing georeferenced GeoTIFFs.

The target NDVI marginal is drawn from a two-component Gaussian mixture whose
parameters come from a Latin hypercube sample (see :mod:`nlm_synth.et_params`).
Passing ``--dist normal`` collapses that mixture to a single Gaussian with the
same mean and standard deviation, which is a direct way to ask what is lost by
treating a bimodal surface as unimodal.

Run with::

    python examples/run_geotiff_monte_carlo.py --out-dir outputs/ndvi_mc_geotiff
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nlm_synth.et_params import MixtureETParameter
from nlm_synth.xarray_mc import run_experiments_geotiff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/ndvi_mc_geotiff")
    parser.add_argument("--dist", choices=["mixture", "normal"], default="mixture",
                        help="bimodal mixture, or its unimodal Gaussian approximation")
    parser.add_argument("--size", type=int, default=512, help="field size in cells")
    parser.add_argument("--runs", type=int, default=5, help="realisations per generator")
    parser.add_argument("--pixel-size", type=float, default=30.0, help="cell size in metres")
    parser.add_argument("--crs", default="EPSG:32611", help="output CRS")
    parser.add_argument("--seed", type=int, default=123, help="master random seed")
    args = parser.parse_args()

    # Build the NDVI marginal from a sampled mixture.
    ndvi_param = MixtureETParameter(
        name="NDVI",
        mu1_bounds=(0.1, 0.5),
        mu2_bounds=(0.5, 0.9),
        w1_bounds=(0.2, 0.8),
        sigma1_bounds=(0.02, 0.05),
        sigma2_bounds=(0.02, 0.05),
    )
    ndvi_param.lhs_sample(n_samples=1, seed=args.seed)
    samples = np.clip(ndvi_param.create_dist(dist_type=args.dist).sample(100_000), -0.2, 1.0)

    # Upper-left corner of the grid, in CRS units. To match a real scene instead:
    #   import rioxarray as rxr
    #   template = rxr.open_rasterio('scene.tif').squeeze()
    #   crs = template.rio.crs.to_string()
    #   pixel_size = template.rio.resolution()[0]
    #   x0, y0 = template.rio.transform().c, template.rio.transform().f
    x0, y0 = 500_000.0, 4_000_000.0

    df, meta = run_experiments_geotiff(
        samples=samples,
        out_dir=args.out_dir,
        nrow=args.size,
        ncol=args.size,
        pixel_size=args.pixel_size,
        x0=x0,
        y0=y0,
        crs=args.crs,
        coarsen_factors=(1, 2, 4, 8, 16, 32),
        n_runs=args.runs,
        random_seed=args.seed,
        name_prefix=f"ndvi_{args.dist}",
        progress=True,
    )

    print(f"\n{len(df)} rows and {len(list(Path(args.out_dir).rglob('*.tif')))} rasters "
          f"written to {args.out_dir}")
    print(f"CRS {meta['crs']}, pixel sizes {sorted(df['pixel_size'].unique())} m")
    print("\nMoran's I by generator and pixel size:")
    print(df.pivot_table(index="pixel_size", columns="label", values="morans_I").round(3))


if __name__ == "__main__":
    main()
