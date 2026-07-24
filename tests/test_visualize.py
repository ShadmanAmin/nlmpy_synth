"""Tests for the plotting helpers."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nlm_synth.visualize import plot_field_grid, plot_marginal, plot_metric_by_scale  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def results():
    rows = []
    for label in ("lf", "hf"):
        for run in range(3):
            for factor in (1, 2, 4):
                rows.append({
                    "label": label,
                    "run": run,
                    "factor": factor,
                    "morans_I": 0.9 / factor + 0.01 * run,
                    "variance": 0.05 / factor,
                })
    return pd.DataFrame(rows)


class TestPlotMetricByScale:
    def test_one_line_per_group(self, results):
        _, ax = plot_metric_by_scale(results)
        assert len(ax.get_lines()) == results["label"].nunique()

    def test_plots_the_mean_across_runs(self, results):
        _, ax = plot_metric_by_scale(results, show_spread=False)
        expected = results[results["label"] == "hf"].groupby("factor")["morans_I"].mean()
        line = next(ln for ln in ax.get_lines() if ln.get_label() == "hf")
        np.testing.assert_allclose(line.get_ydata(), expected.to_numpy())

    def test_spread_adds_a_shaded_band(self, results):
        _, with_spread = plot_metric_by_scale(results, show_spread=True)
        _, without = plot_metric_by_scale(results, show_spread=False)
        assert len(with_spread.collections) > len(without.collections)

    def test_labels_and_legend(self, results):
        _, ax = plot_metric_by_scale(results, metric="variance")
        assert "Variance" in ax.get_ylabel()
        assert "Coarsening factor" in ax.get_xlabel()
        assert {t.get_text() for t in ax.get_legend().get_texts()} == {"lf", "hf"}

    def test_draws_onto_a_supplied_axes(self, results):
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot_metric_by_scale(results, ax=ax)
        assert returned_ax is ax and returned_fig is fig

    def test_missing_column_raises_with_a_useful_message(self, results):
        with pytest.raises(KeyError, match="not in DataFrame"):
            plot_metric_by_scale(results, metric="no_such_metric")


class TestPlotFieldGrid:
    def test_one_axes_per_field(self):
        fields = [np.random.default_rng(i).random((8, 8)) for i in range(5)]
        _, axes = plot_field_grid(fields, ncols=3)
        assert len(axes) == 5

    def test_titles_are_applied(self):
        fields = [np.zeros((4, 4)), np.ones((4, 4))]
        _, axes = plot_field_grid(fields, titles=["a", "b"])
        assert [ax.get_title() for ax in axes] == ["a", "b"]

    def test_spare_panels_are_hidden(self):
        fig, axes = plot_field_grid([np.zeros((4, 4))] * 3, ncols=2)
        # A 2x2 grid holding 3 fields leaves one panel, which must be turned off.
        assert len(fig.axes) > len(axes)

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            plot_field_grid([])

    def test_rejects_mismatched_titles(self):
        with pytest.raises(ValueError, match="one entry per field"):
            plot_field_grid([np.zeros((4, 4))], titles=["a", "b"])


class TestPlotMarginal:
    def test_histogram_and_ecdf(self):
        samples = np.random.default_rng(0).normal(size=1_000)
        _, axes = plot_marginal(samples, label="NDVI")
        assert len(axes) == 2
        assert "ECDF" in axes[1].get_title()

    def test_ecdf_is_monotonic_and_spans_zero_to_one(self):
        samples = np.random.default_rng(1).normal(size=500)
        _, axes = plot_marginal(samples)
        y = axes[1].get_lines()[0].get_ydata()
        assert np.all(np.diff(y) >= 0)
        assert y[0] == pytest.approx(0.0) and y[-1] < 1.0

    def test_non_finite_values_are_dropped(self):
        _, _ = plot_marginal(np.array([0.1, np.nan, 0.9, np.inf]))

    def test_rejects_all_non_finite(self):
        with pytest.raises(ValueError, match="no finite values"):
            plot_marginal(np.array([np.nan, np.inf]))
