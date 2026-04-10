"""
Tests for WSITiler.create_tile_file() and the blur metric utilities.
"""

import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from PIL import Image

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_synthetic_tiff(path: Path, color=(180, 120, 130), size=(672, 672)):
    """Save a solid-color RGB TIFF large enough for several 224x224 tiles."""
    img = Image.new("RGB", size, color=color)
    img.save(path, format="TIFF")
    return path


# ── Group A: unit tests for utils blur functions ──────────────────────────────

class TestLaplacianVariance:
    def test_uniform_image_returns_zero(self):
        from tbccsi.utils import laplacian_variance
        arr = np.zeros((224, 224), dtype=np.float32)
        assert laplacian_variance(arr) == 0.0

    def test_noisy_image_returns_positive(self):
        from tbccsi.utils import laplacian_variance
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (224, 224)).astype(np.float32)
        assert laplacian_variance(arr) > 0.0


class TestFftHighFreqEnergy:
    def test_bounds_on_various_inputs(self):
        from tbccsi.utils import fft_high_freq_energy
        rng = np.random.default_rng(0)
        for arr in [
            np.zeros((224, 224), dtype=np.float32),
            np.full((224, 224), 128.0, dtype=np.float32),
            rng.integers(0, 256, (224, 224)).astype(np.float32),
        ]:
            result = fft_high_freq_energy(arr)
            assert 0.0 <= result <= 1.0

    def test_noise_higher_than_smooth(self):
        from tbccsi.utils import fft_high_freq_energy
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 256, (224, 224)).astype(np.float32)
        # Low-frequency sinusoid
        x = np.linspace(0, 2 * np.pi, 224)
        smooth = (np.sin(x[np.newaxis, :]) * 127 + 128).astype(np.float32)
        smooth = np.broadcast_to(smooth, (224, 224)).copy()
        assert fft_high_freq_energy(noise) > fft_high_freq_energy(smooth)

    def test_lower_cutoff_gives_higher_or_equal_energy(self):
        from tbccsi.utils import fft_high_freq_energy
        rng = np.random.default_rng(7)
        arr = rng.integers(0, 256, (224, 224)).astype(np.float32)
        assert fft_high_freq_energy(arr, cutoff_fraction=0.1) >= fft_high_freq_energy(arr, cutoff_fraction=0.5)


# ── Group B: integration tests for create_tile_file() ────────────────────────

@pytest.fixture
def synthetic_tiff(tmp_path):
    return _make_synthetic_tiff(tmp_path / "slide.tif")


@pytest.fixture
def tiler(tmp_path, synthetic_tiff):
    from tbccsi.wsi_tiler import WSITiler
    tile_file = tmp_path / "tiles.csv"
    return WSITiler("SYNTH", synthetic_tiff, tmp_path, tile_file)


class TestCreateTileFile:
    def test_produces_csv(self, tiler, tmp_path):
        tiler.create_tile_file()
        assert (tmp_path / "tiles.csv").exists()

    def test_column_names(self, tiler, tmp_path):
        tiler.create_tile_file()
        df = pd.read_csv(tmp_path / "tiles.csv")
        assert list(df.columns) == ['x', 'y', 'tile_id', 'mean_red', 'mean_green', 'mean_blue', 'lap_var', 'fft_hfe']

    def test_lap_var_non_negative(self, tiler, tmp_path):
        tiler.create_tile_file()
        df = pd.read_csv(tmp_path / "tiles.csv")
        assert (df['lap_var'] >= 0.0).all()

    def test_fft_hfe_in_range(self, tiler, tmp_path):
        tiler.create_tile_file()
        df = pd.read_csv(tmp_path / "tiles.csv")
        assert (df['fft_hfe'] >= 0.0).all()
        assert (df['fft_hfe'] <= 1.0).all()

    def test_tile_id_monotonic(self, tiler, tmp_path):
        tiler.create_tile_file()
        df = pd.read_csv(tmp_path / "tiles.csv")
        assert list(df['tile_id']) == list(range(len(df)))

    def test_skips_existing_file(self, tiler, tmp_path):
        tiler.create_tile_file()
        mtime_before = os.path.getmtime(tmp_path / "tiles.csv")
        tiler.create_tile_file()
        mtime_after = os.path.getmtime(tmp_path / "tiles.csv")
        assert mtime_before == mtime_after

    def test_all_white_image_zero_rows(self, tmp_path):
        from tbccsi.wsi_tiler import WSITiler
        white_tiff = _make_synthetic_tiff(tmp_path / "white.tif", color=(255, 255, 255))
        tile_file = tmp_path / "white_tiles.csv"
        tiler = WSITiler("WHITE", white_tiff, tmp_path, tile_file)
        tiler.create_tile_file()
        df = pd.read_csv(tile_file)
        assert len(df) == 0
        assert list(df.columns) == ['x', 'y', 'tile_id', 'mean_red', 'mean_green', 'mean_blue', 'lap_var', 'fft_hfe']

    def test_fft_cutoff_affects_output(self, tmp_path):
        from tbccsi.wsi_tiler import WSITiler
        tiff_a = _make_synthetic_tiff(tmp_path / "slide_a.tif")
        tiff_b = _make_synthetic_tiff(tmp_path / "slide_b.tif")
        dir_a, dir_b = tmp_path / "a", tmp_path / "b"
        dir_a.mkdir(); dir_b.mkdir()

        t_a = WSITiler("A", tiff_a, dir_a, dir_a / "tiles.csv")
        t_a.create_tile_file(fft_cutoff=0.1)
        t_b = WSITiler("B", tiff_b, dir_b, dir_b / "tiles.csv")
        t_b.create_tile_file(fft_cutoff=0.9)

        mean_a = pd.read_csv(dir_a / "tiles.csv")['fft_hfe'].mean()
        mean_b = pd.read_csv(dir_b / "tiles.csv")['fft_hfe'].mean()
        assert mean_a != mean_b

    def test_missing_tile_file_path_raises(self, tmp_path, synthetic_tiff):
        from tbccsi.wsi_tiler import WSITiler
        tiler = WSITiler("SYNTH", synthetic_tiff, tmp_path, None)
        with pytest.raises(ValueError):
            tiler.create_tile_file()


# ── Group C: smoke test against real image ────────────────────────────────────

REAL_SLIDE = Path('/users/dgibbs/tmp/tbccsi_tests/test_img_1.tif')

@pytest.mark.skipif(not REAL_SLIDE.exists(), reason="real test image not available")
def test_create_tile_file_real_image(tmp_path):
    from tbccsi.wsi_tiler import WSITiler
    tile_file = tmp_path / "real_tiles.csv"
    tiler = WSITiler("TEST1", REAL_SLIDE, tmp_path, tile_file)
    tiler.create_tile_file()
    df = pd.read_csv(tile_file)
    print(f"Real image: {len(df)} tiles found")
    assert list(df.columns) == ['x', 'y', 'tile_id', 'mean_red', 'mean_green', 'mean_blue', 'lap_var', 'fft_hfe']


# ── CLI tests ─────────────────────────────────────────────────────────────────

class TestTileCli:
    def test_tile_command_runs(self, tmp_path):
        from typer.testing import CliRunner
        from tbccsi.cli import app
        tiff = _make_synthetic_tiff(tmp_path / "slide.tif")
        runner = CliRunner()
        result = runner.invoke(app, [
            "tile",
            "--sample-id", "SYNTH",
            "--input-slide", str(tiff),
            "--work-dir", str(tmp_path),
            "--tile-file", "tiles.csv",
            "--fft-cutoff", "0.5",
        ])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "tiles.csv").exists()

    def test_tile_command_invalid_fft_cutoff(self, tmp_path):
        from typer.testing import CliRunner
        from tbccsi.cli import app
        tiff = _make_synthetic_tiff(tmp_path / "slide.tif")
        runner = CliRunner()
        result = runner.invoke(app, [
            "tile",
            "--sample-id", "SYNTH",
            "--input-slide", str(tiff),
            "--work-dir", str(tmp_path),
            "--tile-file", "tiles.csv",
            "--fft-cutoff", "not_a_float",
        ])
        assert result.exit_code != 0
