"""Figures showing how each Perlin parameter changes the synthetic landscape.

Produces a 4x4 overview (one row per parameter) plus one 2xN figure per
parameter contrasting the raw Perlin field with its NDVI-mapped counterpart.
All parameters other than the one being swept are held at the baseline below.

Run with::

    python examples/plot_parameter_sweeps.py --out-dir docs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nlm_synth import perlin_field, plot_field_grid, rank_map_to_distribution

BASELINE = dict(periods=(4, 4), octaves=5, lacunarity=2, persistence=0.6)

SWEEPS = {
    "periods": [(2, 2), (3, 3), (4, 4), (6, 6)],
    "octaves": [1, 2, 3, 5],
    "lacunarity": [1, 2, 3, 4],
    "persistence": [0.3, 0.5, 0.7, 0.9],
}

# What each parameter controls, for the figure captions.
DESCRIPTIONS = {
    "periods": "patch size: more periods means finer patches",
    "octaves": "number of noise layers: more octaves adds fine detail",
    "lacunarity": "frequency step between octaves",
    "persistence": "amplitude decay between octaves: higher keeps more fine detail",
}


def label_for(name: str, value) -> str:
    return f"{name}={value[0]}x{value[1]}" if name == "periods" else f"{name}={value}"


def sweep_fields(name: str, values, size: int, seed_base: int) -> list[np.ndarray]:
    """One field per value of ``name``, all other parameters at the baseline."""
    fields = []
    for offset, value in enumerate(values):
        kwargs = {**BASELINE, name: value}
        fields.append(perlin_field(size, size, seed=seed_base + offset, **kwargs))
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="docs/figures")
    parser.add_argument("--size", type=int, default=256, help="preview size in cells")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--dpi", type=int, default=110, help="figure resolution")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    samples = np.clip(
        np.hstack([rng.normal(0.70, 0.08, 50_000), rng.normal(0.20, 0.09, 30_000)]), -0.2, 1.0
    )

    # 4x4 overview: one row per parameter, raw fields.
    all_fields, all_titles = [], []
    for row, (name, values) in enumerate(SWEEPS.items()):
        all_fields.extend(sweep_fields(name, values, args.size, args.seed + 100 * row))
        all_titles.extend(label_for(name, value) for value in values)

    fig, _ = plot_field_grid(all_fields, titles=all_titles, ncols=4, cmap="plasma")
    fig.suptitle("Perlin parameter sweeps (raw fields)", y=1.0)
    fig.savefig(out_dir / "perlin_parameter_sweeps.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Same grid, quantile-mapped onto the NDVI marginal.
    mapped = [rank_map_to_distribution(field, samples) for field in all_fields]
    fig, _ = plot_field_grid(mapped, titles=all_titles, ncols=4, cmap="plasma")
    fig.suptitle("Perlin parameter sweeps (mapped to the NDVI distribution)", y=1.0)
    fig.savefig(out_dir / "perlin_parameter_sweeps_ndvi_mapped.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Per-parameter detail: raw on the top row, NDVI-mapped below.
    for index, (name, values) in enumerate(SWEEPS.items(), start=1):
        raw = sweep_fields(name, values, args.size, args.seed + 100 * (index - 1))
        fields = raw + [rank_map_to_distribution(field, samples) for field in raw]
        titles = [f"{label_for(name, v)} (raw)" for v in values]
        titles += [f"{label_for(name, v)} (NDVI)" for v in values]

        fig, _ = plot_field_grid(fields, titles=titles, ncols=len(values), cmap="viridis")
        fig.suptitle(f"{name}: {DESCRIPTIONS[name]}", y=1.0)
        fig.savefig(
            out_dir / f"fig{index}_{name}_perlin_vs_ndvi.png", dpi=args.dpi, bbox_inches="tight"
        )
        plt.close(fig)

    print(f"Figures written to {out_dir}")


if __name__ == "__main__":
    main()
