"""
Unit tests for map plotting helpers.

Run with: pytest tests/test_visualization.py -v
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.visualization import plot_predictions  # noqa: E402
from src.visualization.plot_predictions import (  # noqa: E402
    plot_prediction_comparison,
    recenter_longitudes,
)

# WeatherBench2 64x32 grid: longitude centres 0, 5.625, ..., 354.375 °E
H, W = 32, 64
DLON = 360.0 / W


class TestRecenterLongitudes:
    """Tests for drawing 0..360°E grids on a [-180, 180] map."""

    def test_shape_and_extent(self):
        out, extent = recenter_longitudes(np.zeros((4, H, W)))
        assert out.shape == (4, H, 2 * W)
        assert extent == [-180.0, 180.0, -90.0, 90.0]

    def test_cells_drawn_at_their_true_longitude(self):
        """Every pixel must show the grid cell that covers its map position."""
        centres = np.arange(W) * DLON
        field = np.broadcast_to(centres, (H, W))

        out, extent = recenter_longitudes(field)

        pixel_width = (extent[1] - extent[0]) / out.shape[-1]
        pixel_centres = extent[0] + (np.arange(out.shape[-1]) + 0.5) * pixel_width
        # Signed distance to the plotted cell's centre, wrapped to [-180, 180)
        offset = (pixel_centres - out[0] + 180.0) % 360.0 - 180.0
        assert np.all(np.abs(offset) < DLON / 2)

    def test_greenwich_is_in_the_middle(self):
        """The 0°E column must land at the centre of the map, not at 180°W."""
        field = np.zeros((H, W))
        field[:, 0] = 1.0  # cell centred on 0°E

        out, _ = recenter_longitudes(field)

        marked = np.flatnonzero(out[0])
        assert list(marked) == [W - 1, W]

    def test_odd_width_raises(self):
        with pytest.raises(ValueError):
            recenter_longitudes(np.zeros((H, W - 1)))


class TestPlotPredictionComparison:
    """Smoke test for the comparison figure (no cartopy, no network)."""

    def test_saves_figure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plot_predictions, "HAS_CARTOPY", False)
        x = np.random.randn(4, H, W)
        save_path = tmp_path / "prediction.png"

        fig = plot_prediction_comparison(
            x, x + 0.1, x,
            variable_idx=0,
            variable_names=["t850", "z500", "u10", "v10"],
            save_path=str(save_path),
            colorbar_label="K",
        )

        assert save_path.exists()
        plt.close(fig)
