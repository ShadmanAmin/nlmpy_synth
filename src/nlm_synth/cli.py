"""Command-line interface: ``nlm-synth <command>``.

Provides one-command reproduction of the two experiments and the parameter fit,
so results can be regenerated without writing any Python.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_samples(path: str | None, seed: int) -> np.ndarray:
    """Load the target marginal from a file, or build the demo bimodal sample.

    Accepts ``.npy`` arrays, single-column text/CSV files, or any raster
    readable by rioxarray. With no path, returns a synthetic vegetation/soil
    mixture so the commands are runnable out of the box.
    """
    if path is None:
        rng = np.random.default_rng(seed)
        veg = rng.normal(0.70, 0.08, size=50_000)
        soil = rng.normal(0.20, 0.09, size=30_000)
        return np.clip(np.hstack([veg, soil]), -0.2, 1.0)

    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        values = np.load(path)
    elif suffix in {".csv", ".txt", ".dat"}:
        values = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    else:
        import rioxarray as rxr

        with rxr.open_rasterio(path, masked=True) as raster:
            values = raster.squeeze().values

    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def _cmd_mc(args: argparse.Namespace) -> int:
    import matplotlib

    from .monte_carlo import run_experiments
    from .visualize import plot_metric_by_scale

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(args.samples, args.seed)
    df, meta = run_experiments(
        samples,
        nrow=args.size,
        ncol=args.size,
        coarsen_factors=args.factors,
        n_runs=args.runs,
        semivar=args.semivariogram,
        random_seed=args.seed,
        progress=True,
    )

    df.to_csv(out_dir / "results_mc.csv", index=False)
    (out_dir / "meta_mc.json").write_text(json.dumps(meta, indent=2, default=str))

    for metric in ("morans_I", "variance"):
        fig, _ = plot_metric_by_scale(df, metric=metric)
        fig.savefig(out_dir / f"{metric}_vs_scale.png", dpi=150)
        plt.close(fig)

    print(f"[done] {len(df)} rows -> {out_dir}")
    return 0


def _cmd_geotiff(args: argparse.Namespace) -> int:
    from .xarray_mc import run_experiments_geotiff

    samples = _load_samples(args.samples, args.seed)
    df, meta = run_experiments_geotiff(
        samples,
        out_dir=args.out_dir,
        nrow=args.size,
        ncol=args.size,
        pixel_size=args.pixel_size,
        x0=args.x0,
        y0=args.y0,
        crs=args.crs,
        coarsen_factors=args.factors,
        n_runs=args.runs,
        random_seed=args.seed,
        write_rasters=not args.no_rasters,
        progress=True,
    )
    Path(args.out_dir, "meta_mc_geotiff.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )
    print(f"[done] {len(df)} rows -> {args.out_dir}")
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    from .approximations import fit_perlin_parameters_geotiff

    limit = args.max_internal_dim
    if limit != "auto":
        limit = int(limit)

    best = fit_perlin_parameters_geotiff(
        in_tif=args.raster,
        out_csv=args.out_csv,
        n_repeats=args.repeats,
        seed=args.seed,
        max_internal_dim=limit,
        save_diagnostics_csv=args.diagnostics_csv,
        band=args.band,
        verbose=True,
    )
    print(json.dumps(best, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nlm-synth",
        description="Neutral landscape models with prescribed marginal distributions.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--samples",
            help="path to a .npy/.csv/.txt array or a raster defining the target "
            "marginal; omit to use a synthetic bimodal NDVI demo",
        )
        p.add_argument("--size", type=int, default=512, help="field size in cells (default: 512)")
        p.add_argument("--runs", type=int, default=10, help="realisations per generator (default: 10)")
        p.add_argument(
            "--factors",
            type=int,
            nargs="+",
            default=[1, 2, 4, 8, 16, 32],
            help="coarsening factors (default: 1 2 4 8 16 32)",
        )
        p.add_argument("--seed", type=int, default=42, help="master random seed (default: 42)")

    mc = sub.add_parser("mc", help="run the NumPy Monte Carlo experiment and plot the results")
    add_common(mc)
    mc.add_argument("--out-dir", default="outputs", help="output directory (default: outputs)")
    mc.add_argument(
        "--semivariogram", action="store_true", help="also compute semivariogram range and sill"
    )
    mc.set_defaults(func=_cmd_mc)

    geo = sub.add_parser("geotiff", help="run the Monte Carlo experiment writing per-scale GeoTIFFs")
    add_common(geo)
    geo.add_argument("--out-dir", default="outputs/ndvi_mc_geotiff", help="output directory")
    geo.add_argument("--pixel-size", type=float, default=30.0, help="cell size in CRS units")
    geo.add_argument("--x0", type=float, default=500_000.0, help="upper-left x coordinate")
    geo.add_argument("--y0", type=float, default=4_000_000.0, help="upper-left y coordinate")
    geo.add_argument("--crs", default="EPSG:32611", help="output CRS (default: EPSG:32611)")
    geo.add_argument(
        "--no-rasters", action="store_true", help="compute statistics without writing GeoTIFFs"
    )
    geo.set_defaults(func=_cmd_geotiff)

    fit = sub.add_parser("fit", help="fit Perlin parameters to an observed raster")
    fit.add_argument("raster", help="input GeoTIFF")
    fit.add_argument("--out-csv", default="best_params.csv", help="where to write the best parameters")
    fit.add_argument("--diagnostics-csv", help="where to write every candidate score")
    fit.add_argument("--band", type=int, default=1, help="1-based band index (default: 1)")
    fit.add_argument("--repeats", type=int, default=1, help="realisations averaged per candidate")
    fit.add_argument("--seed", type=int, default=1234, help="base random seed (default: 1234)")
    fit.add_argument(
        "--max-internal-dim",
        default="auto",
        help="skip candidates needing a larger internal nlmpy grid; 'auto' (the "
        "default) uses the raster's own size, an integer widens the search",
    )
    fit.set_defaults(func=_cmd_fit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(__version__)
        return 0
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
