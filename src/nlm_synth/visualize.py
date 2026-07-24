"""Plotting helpers for Monte Carlo results and synthetic fields."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["plot_metric_by_scale", "plot_field_grid", "plot_marginal"]

#: Axis labels for the metrics produced by :func:`~nlm_synth.stats.summarize_stats`.
_METRIC_LABELS = {
    "morans_I": "Moran's I",
    "variance": "Variance",
    "std_dev": "Standard deviation",
    "mean": "Mean",
    "semivar_range": "Semivariogram range (cells)",
    "semivar_sill": "Semivariogram sill",
}


def plot_metric_by_scale(
    df: pd.DataFrame,
    metric: str = "morans_I",
    by: str = "label",
    x: str = "factor",
    show_spread: bool = True,
    ax: plt.Axes | None = None,
):
    """Plot a metric against coarsening scale, one line per generator.

    Parameters
    ----------
    df:
        Tidy results from :func:`~nlm_synth.monte_carlo.run_experiments`.
    metric:
        Column to plot on the y axis.
    by:
        Column that separates the lines, normally the generator ``label``.
    x:
        Column for the x axis; ``'factor'`` for relative scale or
        ``'pixel_size'`` for ground units when available.
    show_spread:
        Shade +/- one standard deviation across runs around each mean line.
    ax:
        Existing axes to draw on. A new figure is created if omitted.

    Returns
    -------
    (fig, ax)
    """
    for column in (metric, by, x):
        if column not in df.columns:
            raise KeyError(f"column {column!r} not in DataFrame; have {list(df.columns)}")

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6, 4))

    for key, sub in df.groupby(by, sort=True):
        grouped = sub.groupby(x)[metric]
        means = grouped.mean()
        line, = ax.plot(means.index, means.to_numpy(), marker="o", label=str(key))
        if show_spread:
            spread = grouped.std().fillna(0.0).to_numpy()
            ax.fill_between(
                means.index,
                means.to_numpy() - spread,
                means.to_numpy() + spread,
                alpha=0.15,
                color=line.get_color(),
                linewidth=0,
            )

    ax.set_xlabel("Coarsening factor (block size)" if x == "factor" else x.replace("_", " "))
    ax.set_ylabel(_METRIC_LABELS.get(metric, metric.replace("_", " ")))
    ax.set_title(f"{_METRIC_LABELS.get(metric, metric)} vs. scale")
    ax.grid(True, ls=":", lw=0.6)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_field_grid(
    fields: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    ncols: int | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
):
    """Show a set of 2-D fields as an image grid.

    Parameters
    ----------
    fields:
        2-D arrays to display.
    titles:
        One title per field.
    ncols:
        Columns in the grid. Defaults to a roughly square layout.
    cmap:
        Matplotlib colormap.
    colorbar:
        Draw a colorbar beside each panel.

    Returns
    -------
    (fig, axes)
        ``axes`` is a flat array with one entry per field.
    """
    fields = list(fields)
    if not fields:
        raise ValueError("fields is empty")
    if titles is not None and len(titles) != len(fields):
        raise ValueError("titles must have one entry per field")

    ncols = ncols or int(np.ceil(np.sqrt(len(fields))))
    nrows = int(np.ceil(len(fields) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), squeeze=False)
    flat = axes.ravel()

    for idx, field in enumerate(fields):
        image = flat[idx].imshow(field, origin="upper", cmap=cmap)
        if titles is not None:
            flat[idx].set_title(titles[idx], fontsize=9)
        flat[idx].set_xticks([])
        flat[idx].set_yticks([])
        if colorbar:
            fig.colorbar(image, ax=flat[idx], fraction=0.046, pad=0.04)

    for spare in flat[len(fields):]:
        spare.axis("off")

    fig.tight_layout()
    return fig, flat[: len(fields)]


def plot_marginal(samples: np.ndarray, bins: int = 60, label: str = "NDVI"):
    """Plot the histogram and empirical CDF of a 1-D sample.

    Returns
    -------
    (fig, axes)
    """
    values = np.asarray(samples, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("samples contains no finite values")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(values, bins=bins, edgecolor="none", alpha=0.9)
    axes[0].set_title(f"{label} distribution")
    axes[0].set_xlabel(label)
    axes[0].set_ylabel("Count")

    ordered = np.sort(values)
    axes[1].plot(ordered, np.linspace(0, 1, ordered.size, endpoint=False), lw=1.5)
    axes[1].set_title(f"{label} ECDF")
    axes[1].set_xlabel(label)
    axes[1].set_ylabel("F(x)")

    for ax in axes:
        ax.grid(True, ls=":", lw=0.6)
    fig.tight_layout()
    return fig, axes
