"""Multi-scale Monte Carlo experiment on a synthetic bimodal NDVI distribution.

Generates fields with four different spatial structures but an identical
marginal distribution, coarsens each to a range of pixel sizes, and plots how
Moran's I and variance respond.

Run with::

    python examples/run_monte_carlo.py --out-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nlm_synth import plot_marginal, plot_metric_by_scale, run_experiments


def bimodal_ndvi_samples(seed: int = 123, n_veg: int = 50_000, n_soil: int = 30_000):
    """A stand-in for real NDVI pixels: a vegetation mode and a bare-soil mode.

    Replace this with your own data to drive the experiment from a real scene::

        samples = np.load('ndvi_samples.npy')
    """
    rng = np.random.default_rng(seed)
    vegetation = rng.normal(0.70, 0.08, size=n_veg)
    soil = rng.normal(0.20, 0.09, size=n_soil)
    return np.clip(np.hstack([vegetation, soil]), -0.2, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs", help="where to write results")
    parser.add_argument("--size", type=int, default=512, help="field size in cells")
    parser.add_argument("--runs", type=int, default=10, help="realisations per generator")
    parser.add_argument("--seed", type=int, default=44, help="master random seed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = bimodal_ndvi_samples()
    fig, _ = plot_marginal(samples, label="NDVI")
    fig.savefig(out_dir / "input_marginal.png", dpi=150)
    plt.close(fig)

    df, meta = run_experiments(
        samples,
        nrow=args.size,
        ncol=args.size,
        coarsen_factors=(1, 2, 4, 8, 16, 32),
        n_runs=args.runs,
        random_seed=args.seed,
        progress=True,
    )
    df.to_csv(out_dir / "results_mc.csv", index=False)

    for metric in ("morans_I", "variance"):
        fig, _ = plot_metric_by_scale(df, metric=metric)
        fig.savefig(out_dir / f"{metric}_vs_scale.png", dpi=150)
        plt.close(fig)

    print(f"\n{len(df)} rows written to {out_dir / 'results_mc.csv'}")
    print(f"generators: {[cfg['label'] for cfg in meta['generator_grid']]}")
    print("\nMoran's I by generator and scale:")
    print(df.pivot_table(index="factor", columns="label", values="morans_I").round(3))


if __name__ == "__main__":
    main()
