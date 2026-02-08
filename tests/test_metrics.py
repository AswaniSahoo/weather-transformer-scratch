"""
Unit tests for evaluation metrics.

Run with: pytest tests/test_metrics.py -v
"""

import pytest
import torch

from src.evaluation.metrics import (
    rmse,
    rmse_per_variable,
    mae,
    anomaly_correlation_coefficient,
    persistence_baseline,
    compute_all_metrics,
)


# Test constants
B, C, H, W = 4, 4, 32, 64


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
