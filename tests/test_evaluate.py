"""
Tests for the evaluation script: the ACC climatology, regional scores, and an
end-to-end run on tiny synthetic data.

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
from src.evaluation.regions import REGIONS, region_cells  # noqa: E402
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
TWO_YEARS = 4 * 365 * 2  # 6-hourly samples in 2017-2018 (no leap day)


def six_hourly(start: str, n: int) -> np.ndarray:
    """n valid times, 6 hours apart, starting at `start`."""
    return np.datetime64(start, "ns") + np.arange(n) * np.timedelta64(6, "h")


@pytest.fixture
def workspace(tmp_path):
    """Processed data, config, and checkpoint for a tiny model."""
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    # The test samples straddle 30 September / 1 October: 2 in the monsoon, 4 after it
    splits = [("train", 12, "2017-09-29T06"), ("val", 4, "2018-09-29T06"),
              ("test", 6, "2019-09-30T12")]
    for split, n, start in splits:
        x = rng.normal(size=(n, 4, H, W)).astype(np.float32)
        np.save(data_dir / f"{split}_inputs.npy", x)
        np.save(data_dir / f"{split}_targets.npy", x + 0.1)
        np.save(data_dir / f"{split}_times.npy", six_hourly(start, n))
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
    assert "day of year" in results["settings"]["acc_climatology"]
    assert figure.exists()

    india = results["regions"]["india"]
    assert india["monsoon_jjas"]["samples"] == 2
    assert india["rest_of_year"]["samples"] == 4
    assert india["rest_of_year"]["persistence"]["rmse_t850_physical"] == pytest.approx(
        0.1 * STATS["t850"]["std"], rel=1e-4
    )


def test_regional_scores_only_see_their_region_and_season():
    times = six_hourly("2019-09-30T00", 8)  # 4 in September, 4 in October
    targets = torch.zeros(8, 1, 32, 64)
    predictions = torch.zeros(8, 1, 32, 64)
    rows, cols = region_cells(REGIONS["india"], 32, 64)
    outside = torch.ones(32, 64, dtype=torch.bool)
    outside[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = False
    predictions[:, :, outside] = 5.0  # wrong everywhere outside India
    predictions[4:, :, ~outside] = 1.0  # wrong over India, but only in October

    india = evaluate.regional_scores(
        predictions, predictions, targets, times, torch.zeros_like(targets), ["t850"]
    )["india"]

    assert india["cells"] == 36
    assert india["monsoon_jjas"]["samples"] == 4
    assert india["monsoon_jjas"]["model"]["rmse"] == 0.0
    assert india["rest_of_year"]["model"]["rmse"] == pytest.approx(1.0)


def test_climatology_uses_the_same_hour_of_day():
    train_times = six_hourly("2017-01-01T00", TWO_YEARS)
    hour = (np.arange(TWO_YEARS) % 4) * 6.0
    train_targets = hour.reshape(-1, 1, 1, 1)

    climatology = evaluate.seasonal_climatology(
        train_targets, train_times, six_hourly("2019-06-01T06", 2)
    )

    assert climatology.shape == (2, 1, 1, 1)
    assert climatology.flatten().tolist() == [6.0, 12.0]


def test_climatology_averages_the_days_around_the_same_date():
    train_times = six_hourly("2017-01-01T00", TWO_YEARS)
    day_of_year = (np.arange(TWO_YEARS) // 4) % 365 + 1.0
    train_targets = day_of_year.reshape(-1, 1, 1, 1)

    climatology = evaluate.seasonal_climatology(
        train_targets, train_times, six_hourly("2019-07-02T00", 1), window_days=30
    )

    # 2 July is day 183; days 153 to 213 average to 183
    assert climatology.item() == pytest.approx(183.0)


def test_climatology_window_wraps_around_new_year():
    train_times = six_hourly("2017-01-01T00", TWO_YEARS)
    in_december = train_times.astype("datetime64[M]").astype(int) % 12 == 11
    train_targets = in_december.astype(np.float64).reshape(-1, 1, 1, 1)

    climatology = evaluate.seasonal_climatology(
        train_targets, train_times, six_hourly("2019-01-05T00", 1), window_days=30
    )

    # The window around 5 January reaches back to 6-31 December: 26 of its 61 days
    assert climatology.item() == pytest.approx(26 / 61)


def test_missing_times_file_says_how_to_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocessing"):
        evaluate.load_target_times(str(tmp_path), "train")
