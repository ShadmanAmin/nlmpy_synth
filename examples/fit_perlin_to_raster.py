"""Estimate the Perlin parameters that best reproduce a real raster's structure.

Given an observed scene, this grid-searches Perlin parameters by matching the
radially averaged power spectrum and Moran's I of the rank-transformed image,
then synthesises a field with the winning parameters and the scene's own value
distribution, so the two can be compared side by side.

Run with::

    python examples/fit_perlin_to_raster.py scene.tif --out-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray as rxr

from nlm_synth import morans_i, perlin_field, rank_map_to_distribution
from nlm_synth.approximations import fit_perlin_parameters_geotiff, square_crop_dataarray


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raster", help="input GeoTIFF, e.g. an NDVI scene")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--band", type=int, default=1, help="1-based band index")
    parser.add_argument("--repeats", type=int, default=1,
                        help="realisations averaged per candidate (reduces noise)")
    parser.add_argument("--max-internal-dim", default="auto",
                        help="skip candidates needing a larger internal nlmpy grid; "
                             "'auto' uses the raster's own size")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = fit_perlin_parameters_geotiff(
        in_tif=args.raster,
        out_csv=str(out_dir / "best_params.csv"),
        save_diagnostics_csv=str(out_dir / "all_param_scores.csv"),
        band=args.band,
        n_repeats=args.repeats,
        seed=args.seed,
        max_internal_dim=args.max_internal_dim if args.max_internal_dim == "auto"
        else int(args.max_internal_dim),
        verbose=True,
    )

    print("\nBest parameters:")
    for key, value in best.items():
        print(f"  {key:14s} {value}")

    scores = pd.read_csv(out_dir / "all_param_scores.csv").sort_values("score")
    print(f"\nTop 5 of {len(scores)} candidates:")
    print(scores.head(5).to_string(index=False))

    # Synthesise a field with the fitted structure and the scene's own marginal.
    with rxr.open_rasterio(args.raster, masked=True) as raster:
        band = raster.sel(band=args.band) if "band" in raster.dims else raster
        observed = square_crop_dataarray(band.squeeze()).values.astype(float)

    samples = observed[np.isfinite(observed)]
    synthetic = rank_map_to_distribution(
        perlin_field(
            *observed.shape,
            periods=best["periods"],
            octaves=best["octaves"],
            lacunarity=best["lacunarity"],
            persistence=best["persistence"],
            seed=args.seed,
        ),
        samples,
    )

    print(f"\nMoran's I  observed={morans_i(observed):.4f}  synthetic={morans_i(synthetic):.4f}")

    vmin, vmax = np.nanpercentile(observed, [2, 98])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, image, title in zip(axes, (observed, synthetic), ("Observed", "Synthetic"), strict=True):
        handle = ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"periods={best['periods']}, octaves={best['octaves']}, "
        f"lacunarity={best['lacunarity']}, persistence={best['persistence']}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "observed_vs_synthetic.png", dpi=150)
    print(f"\nWrote {out_dir / 'observed_vs_synthetic.png'}")


if __name__ == "__main__":
    main()
