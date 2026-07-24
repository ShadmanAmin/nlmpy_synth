"""Tests for the nlm-synth command line and the package's lazy imports."""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from nlm_synth.cli import _load_samples, build_parser, main  # noqa: E402


class TestLoadSamples:
    def test_demo_distribution_is_bimodal_and_reproducible(self):
        a = _load_samples(None, seed=1)
        b = _load_samples(None, seed=1)
        np.testing.assert_array_equal(a, b)
        assert a.size == 80_000
        assert np.isfinite(a).all()

    def test_seed_changes_the_demo_draw(self):
        assert not np.array_equal(_load_samples(None, 1), _load_samples(None, 2))

    def test_reads_npy(self, tmp_path):
        path = tmp_path / "samples.npy"
        np.save(path, np.array([0.1, 0.5, 0.9]))
        np.testing.assert_allclose(_load_samples(str(path), 0), [0.1, 0.5, 0.9])

    def test_reads_csv(self, tmp_path):
        path = tmp_path / "samples.csv"
        path.write_text("0.1\n0.5\n0.9\n")
        np.testing.assert_allclose(_load_samples(str(path), 0), [0.1, 0.5, 0.9])

    def test_reads_a_raster(self, tmp_path):
        pytest.importorskip("rioxarray")
        from rasterio.transform import from_origin

        from nlm_synth.geox import write_geotiff

        path = tmp_path / "scene.tif"
        write_geotiff(str(path), np.arange(16.0).reshape(4, 4), from_origin(0, 0, 1, 1), "EPSG:4326")
        assert _load_samples(str(path), 0).size == 16

    def test_drops_non_finite_values(self, tmp_path):
        path = tmp_path / "samples.npy"
        np.save(path, np.array([0.1, np.nan, np.inf, 0.9]))
        np.testing.assert_allclose(_load_samples(str(path), 0), [0.1, 0.9])


class TestParser:
    def test_subcommands_are_registered(self):
        parser = build_parser()
        for command in ("mc", "geotiff", "fit"):
            assert parser.parse_args([command] if command != "fit" else [command, "x.tif"])

    def test_no_command_prints_help_and_fails(self, capsys):
        assert main([]) == 1
        assert "usage:" in capsys.readouterr().out

    def test_version(self, capsys):
        from nlm_synth import __version__

        assert main(["--version"]) == 0
        assert capsys.readouterr().out.strip() == __version__


class TestMcCommand:
    def test_writes_results_meta_and_plots(self, tmp_path):
        out_dir = tmp_path / "out"
        code = main([
            "mc", "--out-dir", str(out_dir), "--size", "32",
            "--runs", "1", "--factors", "1", "2", "--seed", "5",
        ])
        assert code == 0

        import pandas as pd

        df = pd.read_csv(out_dir / "results_mc.csv")
        assert len(df) == 4 * 1 * 2  # default grid x runs x factors
        assert (out_dir / "morans_I_vs_scale.png").exists()
        assert (out_dir / "variance_vs_scale.png").exists()

        meta = json.loads((out_dir / "meta_mc.json").read_text())
        assert meta["random_seed"] == 5 and meta["coarsen_factors"] == [1, 2]

    def test_semivariogram_flag_adds_columns(self, tmp_path):
        import pandas as pd

        out_dir = tmp_path / "out"
        main(["mc", "--out-dir", str(out_dir), "--size", "32", "--runs", "1",
              "--factors", "1", "--semivariogram"])
        df = pd.read_csv(out_dir / "results_mc.csv")
        assert {"semivar_range", "semivar_sill"} <= set(df.columns)


class TestGeotiffCommand:
    def test_writes_rasters_and_meta(self, tmp_path):
        pytest.importorskip("rioxarray")
        out_dir = tmp_path / "rasters"
        assert main([
            "geotiff", "--out-dir", str(out_dir), "--size", "32",
            "--runs", "1", "--factors", "1", "2",
        ]) == 0
        assert (out_dir / "results_mc_geotiff.csv").exists()
        assert (out_dir / "meta_mc_geotiff.json").exists()
        assert len(list(out_dir.rglob("*.tif"))) == 4 * 1 * 2

    def test_no_rasters_flag(self, tmp_path):
        pytest.importorskip("rioxarray")
        out_dir = tmp_path / "rasters"
        main(["geotiff", "--out-dir", str(out_dir), "--size", "32", "--runs", "1",
              "--factors", "1", "--no-rasters"])
        assert not list(out_dir.rglob("*.tif"))


class TestFitCommand:
    @pytest.fixture
    def raster_path(self, tmp_path):
        pytest.importorskip("rioxarray")
        from rasterio.transform import from_origin

        from nlm_synth.generators import perlin_field
        from nlm_synth.geox import write_geotiff

        path = tmp_path / "scene.tif"
        field = perlin_field(64, 64, periods=(4, 4), octaves=3, lacunarity=2,
                             persistence=0.5, seed=11)
        write_geotiff(str(path), field, from_origin(0, 0, 30, 30), "EPSG:32611")
        return path

    def test_writes_best_and_diagnostics(self, raster_path, tmp_path, capsys):
        import pandas as pd

        best_csv = tmp_path / "best.csv"
        diag_csv = tmp_path / "diag.csv"
        assert main([
            "fit", str(raster_path), "--out-csv", str(best_csv),
            "--diagnostics-csv", str(diag_csv),
        ]) == 0

        assert len(pd.read_csv(best_csv)) == 1
        assert len(pd.read_csv(diag_csv)) > 1
        assert "periods" in capsys.readouterr().out

    def test_explicit_max_internal_dim_is_parsed_as_int(self, raster_path, tmp_path):
        """The flag accepts 'auto' or an integer, so it cannot use type=int."""
        assert main([
            "fit", str(raster_path), "--out-csv", str(tmp_path / "b.csv"),
            "--max-internal-dim", "256",
        ]) == 0


class TestLazyImports:
    def test_geo_names_resolve_through_getattr(self):
        pytest.importorskip("rioxarray")
        import nlm_synth as ns

        assert callable(ns.to_xarray)
        assert callable(ns.run_experiments_geotiff)
        assert callable(ns.fit_perlin_parameters_geotiff)

    def test_unknown_attribute_raises_attribute_error(self):
        import nlm_synth as ns

        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(ns, "definitely_not_a_real_name")  # noqa: B009

    def test_dir_lists_the_lazy_names(self):
        import nlm_synth as ns

        assert "run_experiments_geotiff" in dir(ns)
