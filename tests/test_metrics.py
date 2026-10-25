"""
Unit tests for evaluation metrics.

Run with: pytest tests/test_metrics.py -v
"""

import pytest
import torch

from src.evaluation.metrics import (
    acc_per_variable,
    latitude_weights,
    rmse,
    rmse_per_variable,
    mae,
    anomaly_correlation_coefficient,
    persistence_baseline,
    compute_all_metrics,
)


# Test constants
B, C, H, W = 4, 4, 32, 64


class TestLatitudeWeights:
    """Tests for cos(latitude) area weights."""

    def test_shape_and_mean(self):
        w = latitude_weights(H)
        assert w.shape == (H,)
        assert abs(w.mean().item() - 1.0) < 1e-6

    def test_matches_cosine_of_wb2_latitudes(self):
        """WeatherBench2 64x32 rows are centred on -87.1875..87.1875 degrees."""
        lat = torch.linspace(-87.1875, 87.1875, H, dtype=torch.float64)
        expected = torch.cos(torch.deg2rad(lat))
        expected = expected / expected.mean()
        assert torch.allclose(latitude_weights(H), expected.float(), atol=1e-6)

    def test_symmetric_and_largest_at_equator(self):
        w = latitude_weights(H)
        assert torch.allclose(w, w.flip(0))
        assert w[H // 2] > w[0] * 10  # polar rows cover far less area


class TestLatitudeWeightedMetrics:
    """Weighted metrics must count each cell by its area."""

    def test_uniform_error_unaffected_by_weights(self):
        pred = torch.ones(1, 1, H, W)
        target = torch.zeros(1, 1, H, W)
        w = latitude_weights(H)
        assert abs(rmse(pred, target, w).item() - 1.0) < 1e-6
        assert abs(mae(pred, target, w).item() - 1.0) < 1e-6

    def test_polar_errors_count_less(self):
        """An error confined to the polar rows is small when area-weighted."""
        pred = torch.zeros(1, 1, H, W)
        pred[:, :, [0, -1], :] = 1.0
        target = torch.zeros(1, 1, H, W)

        unweighted = rmse(pred, target).item()
        weighted = rmse(pred, target, latitude_weights(H)).item()
        assert weighted < unweighted / 3

    def test_per_variable_weighted(self):
        pred = torch.randn(B, C, H, W)
        target = torch.randn(B, C, H, W)
        w = latitude_weights(H)
        per_var = rmse_per_variable(pred, target, w)
        assert per_var.shape == (C,)
        # Same total MSE whether averaged per variable or all at once
        assert abs(per_var.pow(2).mean().item() - rmse(pred, target, w).pow(2).item()) < 1e-5


class TestRMSE:
    """Tests for RMSE metric."""

    def test_identical_inputs_zero(self):
        """RMSE of identical tensors should be 0."""
        x = torch.randn(B, C, H, W)
        assert rmse(x, x).item() < 1e-6

    def test_known_value(self):
        """RMSE of known difference should match expected value."""
        pred = torch.ones(1, 1, 2, 2)
        target = torch.zeros(1, 1, 2, 2)
        assert abs(rmse(pred, target).item() - 1.0) < 1e-6

    def test_positive(self):
        """RMSE should always be non-negative."""
        pred = torch.randn(B, C, H, W)
        target = torch.randn(B, C, H, W)
        assert rmse(pred, target).item() >= 0


class TestRMSEPerVariable:
    """Tests for per-variable RMSE."""

    def test_shape(self):
        """Should return (C,) tensor."""
        pred = torch.randn(B, C, H, W)
        target = torch.randn(B, C, H, W)
        result = rmse_per_variable(pred, target)
        assert result.shape == (C,)

    def test_identical_zero(self):
        """All per-variable RMSE should be 0 for identical tensors."""
        x = torch.randn(B, C, H, W)
        result = rmse_per_variable(x, x)
        assert (result < 1e-6).all()


class TestMAE:
    """Tests for MAE metric."""

    def test_identical_zero(self):
        """MAE of identical tensors should be 0."""
        x = torch.randn(B, C, H, W)
        assert mae(x, x).item() < 1e-6

    def test_known_value(self):
        """MAE of known difference should match."""
        pred = torch.ones(1, 1, 2, 2) * 3
        target = torch.ones(1, 1, 2, 2)
        assert abs(mae(pred, target).item() - 2.0) < 1e-6


class TestACC:
    """Tests for Anomaly Correlation Coefficient."""

    def test_perfect_prediction(self):
        """ACC should be ~1.0 for perfect prediction."""
        target = torch.randn(B, C, H, W)
        acc = anomaly_correlation_coefficient(target, target)
        assert abs(acc.item() - 1.0) < 1e-5

    def test_range(self):
        """ACC should be between -1 and 1."""
        pred = torch.randn(B, C, H, W)
        target = torch.randn(B, C, H, W)
        acc = anomaly_correlation_coefficient(pred, target)
        assert -1.0 <= acc.item() <= 1.0 + 1e-5

    def test_inverted_anomalies_give_minus_one(self):
        climatology = torch.randn(C, H, W)
        anomaly = torch.randn(B, C, H, W)
        acc = anomaly_correlation_coefficient(
            climatology - anomaly, climatology + anomaly,
            climatology=climatology, lat_weights=latitude_weights(H),
        )
        assert abs(acc.item() + 1.0) < 1e-5

    def test_scale_of_anomalies_does_not_matter(self):
        """ACC measures pattern agreement, not amplitude."""
        climatology = torch.randn(C, H, W)
        anomaly = torch.randn(B, C, H, W)
        acc = anomaly_correlation_coefficient(
            climatology + 0.5 * anomaly, climatology + anomaly, climatology=climatology
        )
        assert abs(acc.item() - 1.0) < 1e-5

    def test_per_variable_not_pooled(self):
        """One perfect and one inverted variable average to 0, per variable."""
        climatology = torch.zeros(2, H, W)
        target = torch.randn(B, 2, H, W)
        pred = target.clone()
        pred[:, 1] = -target[:, 1]
        per_var = acc_per_variable(pred, target, climatology)
        assert torch.allclose(per_var, torch.tensor([1.0, -1.0]), atol=1e-5)


class TestPersistenceBaseline:
    """Tests for persistence baseline."""

    def test_returns_dict(self):
        """Should return dict with rmse, mae, acc keys."""
        inputs = torch.randn(B, C, H, W)
        targets = torch.randn(B, C, H, W)
        result = persistence_baseline(inputs, targets)
        assert "rmse" in result
        assert "mae" in result
        assert "acc" in result

    def test_identical_zero_error(self):
        """If input == target (no change), persistence is perfect."""
        x = torch.randn(B, C, H, W)
        result = persistence_baseline(x, x)
        assert result["rmse"] < 1e-6
        assert result["mae"] < 1e-6

    def test_same_keys_as_model_metrics(self):
        """Persistence must be comparable to the model, variable by variable."""
        inputs = torch.randn(B, C, H, W)
        targets = torch.randn(B, C, H, W)
        names = ["t850", "z500", "u10", "v10"]
        w = latitude_weights(H)
        baseline = persistence_baseline(inputs, targets, names, w)
        model = compute_all_metrics(inputs, targets, names, w)
        assert baseline == model


class TestComputeAllMetrics:
    """Tests for compute_all_metrics."""

    def test_returns_all_keys(self):
        """Should contain rmse, mae, acc, and per-var RMSE."""
        pred = torch.randn(B, C, H, W)
        target = torch.randn(B, C, H, W)
        result = compute_all_metrics(pred, target)
        assert "rmse" in result
        assert "mae" in result
        assert "acc" in result

    def test_with_variable_names(self):
        """Per-variable keys should use provided names."""
        pred = torch.randn(B, C, H, W)
        target = torch.randn(B, C, H, W)
        names = ["t850", "z500", "u10", "v10"]
        result = compute_all_metrics(pred, target, variable_names=names)
        assert "rmse_t850" in result
        assert "rmse_v10" in result
        assert "acc_t850" in result
        assert "acc_v10" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
