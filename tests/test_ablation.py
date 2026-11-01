"""
Unit tests for the step 6 ablation script.

Run with: pytest tests/test_ablation.py -v
"""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "ablation", Path(__file__).parent.parent / "scripts" / "ablation.py"
)
ablation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ablation)


def test_variant_config_changes_a_copy():
    base = {
        "model": {"residual": False},
        "loss": {"smoothness_weight": 0.1, "conservation_weight": 0.05},
        "training": {"seed": 1, "checkpoint_dir": "checkpoints"},
    }

    config = ablation.variant_config(base, ablation.VARIANTS["tendency_mse"], 43, Path("runs/x"))

    assert config["model"]["residual"] is True
    assert config["loss"] == {"smoothness_weight": 0.0, "conservation_weight": 0.0}
    assert config["training"] == {"seed": 43, "checkpoint_dir": "runs/x"}
    assert base["model"]["residual"] is False and base["training"]["seed"] == 1


def fake_metrics(value: float) -> dict:
    """metrics.json contents where every model and baseline score equals `value`."""
    scores = {f"rmse_{v}_physical": value for v in ablation.VARIABLES}
    scores.update({f"acc_{v}": value / 10 for v in ablation.VARIABLES})
    entry = {"model": scores, "persistence": scores, "anomaly_persistence": scores}
    return {**entry, "regions": {"india": {"monsoon_jjas": entry}}}


def test_summary_is_mean_and_spread_over_seeds():
    summary = ablation.summarize({"tendency": [fake_metrics(v) for v in (1.0, 2.0, 3.0)]})

    t850 = summary["global"]["tendency"]["rmse_t850_physical"]
    assert summary["global"]["tendency"]["seeds"] == 3
    assert t850["mean"] == pytest.approx(2.0)
    assert t850["std"] == pytest.approx(1.0)
    assert summary["india_jjas"]["persistence"]["seeds"] == 0
    assert "2.000 ± 1.000" in ablation.markdown(summary)
