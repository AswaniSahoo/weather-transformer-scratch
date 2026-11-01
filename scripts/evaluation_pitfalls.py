"""
Measure how much common evaluation shortcuts change a forecast's score.

Every number in the pitfall sections of docs/report.md comes from this
script. It scores the persistence forecast on the test set with and without
each shortcut:

1. No area weighting (every grid cell counts the same).
2. One ACC pooled over all variables, including for a forecast whose t850 is
   exactly backwards.
3. Climatology taken from the test year itself instead of the training years.
4. One climatology for the whole year instead of one per day of year and hour.
5. Persistence instead of anomaly persistence as the baseline over India.

It also measures one loss-design problem:

6. The smoothness term penalizes the truth's own gradients, so even a perfect
   forecast pays it.

Usage:
    python scripts/evaluation_pitfalls.py
    python scripts/evaluation_pitfalls.py --data-dir data/processed --output results/evaluation_pitfalls.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.physics_loss import SpatialSmoothnessLoss  # noqa: E402
from src.evaluation.evaluate import (  # noqa: E402
    load_normalization_stats,
    load_target_times,
    regional_scores,
    seasonal_climatology,
)
from src.evaluation.metrics import acc_per_variable, compute_all_metrics, latitude_weights  # noqa: E402
from src.evaluation.regions import REGIONS, SEASONS, months, region_cells  # noqa: E402

NAMES = ["t850", "z500", "u10", "v10"]


def pooled_acc(prediction: torch.Tensor, target: torch.Tensor, climatology: torch.Tensor) -> float:
    """One correlation over all variables, samples and cells (the old definition)."""
    f, o = prediction - climatology, target - climatology
    return ((f * o).sum() / torch.sqrt((f ** 2).sum() * (o ** 2).sum())).item()


def per_variable(values: torch.Tensor) -> dict:
    return {name: round(float(v), 4) for name, v in zip(NAMES, values)}


def main():
    parser = argparse.ArgumentParser(description="Measure evaluation pitfalls")
    parser.add_argument("--data-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--config", default="configs/default.yaml", help="For the loss weights")
    parser.add_argument("--output", default="results/evaluation_pitfalls.json")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        smoothness_weight = yaml.safe_load(f)["loss"]["smoothness_weight"]

    d = Path(args.data_dir)
    inputs = torch.from_numpy(np.load(d / "test_inputs.npy"))
    targets = torch.from_numpy(np.load(d / "test_targets.npy"))
    train_targets = np.load(d / "train_targets.npy", mmap_mode="r")
    train_times = load_target_times(args.data_dir, "train")
    times = load_target_times(args.data_dir, "test")
    stats = load_normalization_stats(args.data_dir)
    n_lat, n_lon = targets.shape[-2:]
    weights = latitude_weights(n_lat)

    seasonal = seasonal_climatology(train_targets, train_times, times)
    seasonal_at_input = seasonal_climatology(train_targets, train_times, times - np.timedelta64(6, "h"))
    yearly = torch.from_numpy(train_targets.mean(axis=0, dtype=np.float64).astype(np.float32))
    test_year = targets.mean(dim=0)

    # 1. Area weighting
    lat = -90.0 + (np.arange(n_lat) + 0.5) * 180.0 / n_lat
    polar = np.abs(lat) > 60
    area = np.cos(np.deg2rad(lat))
    unweighted = compute_all_metrics(inputs, targets, NAMES)
    weighted = compute_all_metrics(inputs, targets, NAMES, weights)
    area_weighting = {
        "polar_rows_share_of_cells": round(float(polar.mean()), 3),
        "polar_rows_share_of_area": round(float(area[polar].sum() / area.sum()), 3),
        "equator_to_polar_cell_area": round(float(area[n_lat // 2 - 1] / area[0]), 1),
        "persistence_rmse_unweighted": round(unweighted["rmse"], 4),
        "persistence_rmse_weighted": round(weighted["rmse"], 4),
    }

    # 2. Pooled ACC, for persistence and for persistence with t850 reversed
    reversed_t850 = inputs.clone()
    reversed_t850[:, 0] = yearly[0] - (inputs[:, 0] - yearly[0])
    pooling = {
        "persistence_pooled": round(pooled_acc(inputs, targets, yearly), 3),
        "persistence_per_variable": per_variable(acc_per_variable(inputs, targets, yearly)),
        "reversed_t850_pooled": round(pooled_acc(reversed_t850, targets, yearly), 3),
        "reversed_t850_per_variable": per_variable(acc_per_variable(reversed_t850, targets, yearly)),
        "share_of_anomaly_size": per_variable(
            ((targets - yearly) ** 2).sum(dim=(0, 2, 3)) / ((targets - yearly) ** 2).sum()
        ),
    }

    # 3 and 4. Which climatology: test year, training years (one per year), or seasonal
    climatology = {
        name: per_variable(acc_per_variable(inputs, targets, clim, weights))
        for name, clim in [("test_year_mean", test_year), ("training_years_mean", yearly),
                           ("training_years_seasonal", seasonal)]
    }

    # 5. Persistence vs anomaly persistence over India, monsoon season
    rows, cols = (torch.as_tensor(i) for i in region_cells(REGIONS["india"], n_lat, n_lon))
    monsoon = torch.as_tensor(np.isin(months(times), SEASONS["monsoon_jjas"]))
    std = torch.tensor([stats[n]["std"] for n in NAMES]).view(1, -1, 1, 1)

    def india_rms(x):
        x = (x[monsoon][:, :, rows][:, :, :, cols] * std) ** 2
        w = weights[rows].view(1, 1, -1, 1).expand_as(x)
        return per_variable(torch.sqrt((x * w).sum(dim=(0, 2, 3)) / w.sum(dim=(0, 2, 3))))

    india = regional_scores(inputs, inputs, targets, times, seasonal, NAMES, seasonal_at_input, stats)
    india_jjas = india["india"]["monsoon_jjas"]
    daily_cycle = {
        "anomaly_size_rms": india_rms(targets - seasonal),
        "usual_6h_change_rms": india_rms(seasonal - seasonal_at_input),
        "persistence_rmse": {n: round(india_jjas["persistence"][f"rmse_{n}_physical"], 4) for n in NAMES},
        "persistence_acc": {n: round(india_jjas["persistence"][f"acc_{n}"], 4) for n in NAMES},
        "anomaly_persistence_rmse": {
            n: round(india_jjas["anomaly_persistence"][f"rmse_{n}_physical"], 4) for n in NAMES
        },
        "anomaly_persistence_acc": {
            n: round(india_jjas["anomaly_persistence"][f"acc_{n}"], 4) for n in NAMES
        },
    }

    # 6. The smoothness term on the validation targets: what a perfect forecast pays
    truth_smoothness = SpatialSmoothnessLoss()(torch.from_numpy(np.load(d / "val_targets.npy")))
    loss_design = {
        "smoothness_of_validation_targets": round(truth_smoothness.item(), 4),
        "smoothness_weight": smoothness_weight,
        "penalty_paid_by_a_perfect_forecast": round(smoothness_weight * truth_smoothness.item(), 4),
    }

    results = {
        "forecast": "persistence (input state as the 6-hour forecast), test set",
        "test_samples": len(times),
        "units": "RMSE in physical units unless named 'normalized'; persistence_rmse_* are normalized",
        "1_area_weighting": area_weighting,
        "2_pooled_acc": pooling,
        "3_4_acc_by_climatology": climatology,
        "5_india_monsoon_daily_cycle": daily_cycle,
        "6_smoothness_term": loss_design,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
