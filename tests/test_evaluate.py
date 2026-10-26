"""
End-to-end test of the evaluation script on tiny synthetic data.

Run with: pytest tests/test_evaluate.py -v
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.evaluation import evaluate  # noqa: E402
from src.models.weather_transformer import WeatherTransformer  # noqa: E402
from src.visualization import plot_predictions  # noqa: E402

H, W = 8, 16
VARIABLES = ["t850", "z500", "u10", "v10"]
STATS = {
    "t850": {"mean": 275.0, "std": 15.0},
    "z500": {"mean": 54000.0, "std": 3400.0},
    "u10": {"mean": 0.0, "std": 5.0},
    "v10": {"mean": 0.2, "std": 4.0},
}


@pytest.fixture
def workspace(tmp_path):
    """Processed data, config, and checkpoint for a tiny model."""
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    for split, n in [("train", 12), ("val", 4), ("test", 6)]:
        x = rng.normal(size=(n, 4, H, W)).astype(np.float32)
        np.save(data_dir / f"{split}_inputs.npy", x)
        np.save(data_dir / f"{split}_targets.npy", x + 0.1)
    (data_dir / "normalization_stats.json").write_text(json.dumps(STATS))

    config = {
        "data": {
            "variables": {name: {"name": name} for name in VARIABLES},
            "lat_size": H,
            "lon_size": W,
        },
        "model": {"patch_size": 4, "embed_dim": 16, "num_heads": 2, "num_layers": 1},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    model = WeatherTransformer(
        img_height=H, img_width=W, patch_size=4, embed_dim=16, num_heads=2, num_layers=1,
    )
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save({"epoch": 1, "model_state_dict": model.state_dict()}, checkpoint_path)

    return tmp_path, data_dir, config_path, checkpoint_path


def test_main_writes_metrics_and_figure(workspace, monkeypatch):
    tmp_path, data_dir, config_path, checkpoint_path = workspace
    output = tmp_path / "metrics.json"
    figure = tmp_path / "forecast.png"
    monkeypatch.setattr(plot_predictions, "HAS_CARTOPY", False)
    monkeypatch.setattr(sys, "argv", [
        "evaluate",
        "--checkpoint", str(checkpoint_path),
        "--config", str(config_path),
        "--data-dir", str(data_dir),
        "--output", str(output),
        "--figure", str(figure),
    ])

    evaluate.main()

    results = json.loads(output.read_text(encoding="utf-8"))
    assert results["settings"]["physical_units"]["z500"] == "m²/s²"
    for name in ("model", "persistence"):
        for var in VARIABLES:
            assert f"rmse_{var}" in results[name]
            assert f"acc_{var}" in results[name]
            # Physical RMSE = normalized RMSE * that variable's std
            assert results[name][f"rmse_{var}_physical"] == pytest.approx(
                results[name][f"rmse_{var}"] * STATS[var]["std"]
            )
    # Targets are inputs + 0.1 everywhere, so persistence RMSE is exactly 0.1
    assert results["persistence"]["rmse"] == pytest.approx(0.1, abs=1e-5)
    assert results["settings"]["test_samples"] == 6
    assert figure.exists()


def test_training_climatology_is_time_mean(workspace):
    _, data_dir, _, _ = workspace
    expected = np.load(data_dir / "train_targets.npy").mean(axis=0)

    climatology = evaluate.training_climatology(str(data_dir))

    assert climatology.shape == (4, H, W)
    np.testing.assert_allclose(climatology.numpy(), expected, rtol=1e-5, atol=1e-6)
