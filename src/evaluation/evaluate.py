"""
Evaluation script for the Weather Transformer.

Loads a trained model, runs it on the test set, and compares
against the persistence baseline.

Metrics follow WeatherBench2 conventions: latitude (area) weighted, and ACC
computed per sample and variable against a climatology from the training
years for the same day of year and hour of day (see seasonal_climatology).

Usage:
    python -m src.evaluation.evaluate
    python -m src.evaluation.evaluate --checkpoint checkpoints/best_model.pt
    python -m src.evaluation.evaluate --figure results/figures/prediction_sample.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from src.data.dataset import WeatherBenchDataset
from src.models.weather_transformer import WeatherTransformer
from src.evaluation.metrics import (
    compute_all_metrics,
    latitude_weights,
    persistence_baseline,
)

# Physical units of each channel (for reporting errors in real units)
UNITS = {"t850": "K", "z500": "m²/s²", "u10": "m/s", "v10": "m/s"}

# Half-width of the day-of-year window the climatology averages over
CLIMATOLOGY_WINDOW_DAYS = 30


def load_model(checkpoint_path: str, config: dict, device: str) -> WeatherTransformer:
    """Load a trained model from checkpoint."""
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    model = WeatherTransformer(
        in_channels=model_cfg.get("in_channels", 4),
        out_channels=model_cfg.get("out_channels", 4),
        img_height=data_cfg.get("lat_size", 32),
        img_width=data_cfg.get("lon_size", 64),
        patch_size=model_cfg.get("patch_size", 4),
        embed_dim=model_cfg.get("embed_dim", 256),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_layers", 6),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        dropout=0.0,  # No dropout at inference
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")

    return model


def load_target_times(data_dir: str, split: str) -> np.ndarray:
    """Valid time of every target in a split, as saved by preprocessing."""
    path = Path(data_dir) / f"{split}_times.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Re-run preprocessing (python -m src.data.preprocessing) "
            "to save the target times the climatology needs."
        )
    return np.load(path)


def _day_of_year(times: np.ndarray) -> np.ndarray:
    """1 for 1 January, up to 366 for 31 December of a leap year."""
    days = times.astype("datetime64[D]")
    return (days - days.astype("datetime64[Y]")).astype(int) + 1


def _hour_of_day(times: np.ndarray) -> np.ndarray:
    hours = times.astype("datetime64[h]")
    return (hours - hours.astype("datetime64[D]")).astype(int)


def seasonal_climatology(
    train_targets: np.ndarray,
    train_times: np.ndarray,
    times: np.ndarray,
    window_days: int = CLIMATOLOGY_WINDOW_DAYS,
) -> torch.Tensor:
    """
    Climatology for ACC: the usual state for each time's day of year and hour.

    For each entry of `times`, averages the training targets at the same hour
    of day whose day of year is within ±window_days, across all training
    years. The window is needed because a few years of data hold only one
    sample per year for any single date and hour.

    Args:
        train_targets: (N_train, C, H, W) training targets.
        train_times: (N_train,) valid times of the training targets.
        times: (N,) valid times to compute the climatology for.
        window_days: Half-width of the day-of-year window.

    Returns:
        Tensor of shape (N, C, H, W) aligned with `times`, in normalized units.
    """
    train_day, train_hour = _day_of_year(train_times), _hour_of_day(train_times)
    means = {}
    climatology = np.empty((len(times),) + train_targets.shape[1:], dtype=np.float32)

    for i, key in enumerate(zip(_day_of_year(times), _hour_of_day(times))):
        if key not in means:
            day, hour = key
            gap = np.abs(train_day - day)
            gap = np.minimum(gap, 365 - gap)  # wraps around New Year: day 365 is next to day 1
            selected = (gap <= window_days) & (train_hour == hour)
            if not selected.any():
                raise ValueError(f"No training targets at {hour:02d} UTC near day {day}")
            means[key] = train_targets[selected].mean(axis=0, dtype=np.float64)
        climatology[i] = means[key]

    return torch.from_numpy(climatology)


def load_normalization_stats(data_dir: str) -> Optional[Dict[str, Dict[str, float]]]:
    """Load per-variable mean/std written by preprocessing, if present."""
    stats_path = Path(data_dir) / "normalization_stats.json"
    if not stats_path.exists():
        return None
    with open(stats_path) as f:
        return json.load(f)


def add_physical_units(
    metrics: Dict[str, float], stats: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Add rmse_<var>_physical: per-variable RMSE converted to physical units.

    Normalization is x_norm = (x - mean) / std per variable, so an RMSE in
    normalized units times that variable's std is the RMSE in physical units.
    """
    physical = {}
    for var, var_stats in stats.items():
        key = f"rmse_{var}"
        if key in metrics:
            physical[f"{key}_physical"] = metrics[key] * var_stats["std"]
    return {**metrics, **physical}


@torch.no_grad()
def predict(
    model: WeatherTransformer,
    dataset: WeatherBenchDataset,
    device: str,
    batch_size: int = 64,
):
    """Run the model over a dataset; returns (inputs, predictions, targets)."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    all_preds = []
    all_targets = []
    all_inputs = []

    for inputs, targets in loader:
        predictions = model(inputs.to(device))

        all_preds.append(predictions.cpu())
        all_targets.append(targets)
        all_inputs.append(inputs)

    return torch.cat(all_inputs), torch.cat(all_preds), torch.cat(all_targets)


def evaluate_model(
    model: WeatherTransformer,
    dataset: WeatherBenchDataset,
    device: str,
    batch_size: int = 64,
    variable_names: Optional[List[str]] = None,
    lat_weights: Optional[torch.Tensor] = None,
    climatology: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on a dataset and compare with persistence baseline.

    Args:
        model: Trained WeatherTransformer.
        dataset: Test dataset.
        device: Device string.
        batch_size: Batch size for evaluation.
        variable_names: List of variable names.
        lat_weights: Optional (H,) area weights (see latitude_weights).
        climatology: Optional (C, H, W) or per-sample (N, C, H, W)
            climatology for ACC.

    Returns:
        Dict with 'model' and 'persistence' metric dicts.
    """
    inputs, predictions, targets = predict(model, dataset, device, batch_size)

    return {
        "model": compute_all_metrics(
            predictions, targets, variable_names, lat_weights, climatology
        ),
        "persistence": persistence_baseline(
            inputs, targets, variable_names, lat_weights, climatology
        ),
    }


def save_forecast_figure(
    model: WeatherTransformer,
    dataset: WeatherBenchDataset,
    device: str,
    save_path: str,
    sample_idx: int,
    variable_names: List[str],
    variable: str,
    stats: Optional[Dict[str, Dict[str, float]]],
):
    """Plot input / prediction / truth / error maps for one test sample."""
    from src.visualization.plot_predictions import plot_prediction_comparison

    var_idx = variable_names.index(variable)
    inputs, target = dataset[sample_idx]
    with torch.no_grad():
        prediction = model(inputs.unsqueeze(0).to(device))[0].cpu()

    fields = [t[var_idx].numpy() for t in (inputs, prediction, target)]
    label = "normalized (z-score)"
    if stats and variable in stats:
        # Back to physical units for a readable map
        mean, std = stats[variable]["mean"], stats[variable]["std"]
        fields = [f * std + mean for f in fields]
        label = UNITS.get(variable, "")

    plot_prediction_comparison(
        *fields,
        variable_names=[variable],
        save_path=save_path,
        title=f"Weather Transformer — 6h Forecast (test sample {sample_idx})",
        colorbar_label=label,
    )


def print_results(results: Dict[str, Dict[str, float]]):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Model':>12} {'Persistence':>12} {'Better?':>10}")
    print("-" * 60)

    model_m = results["model"]
    persist_m = results["persistence"]

    # Overall metrics, then per-variable ones (rmse_t850, acc_t850, ...)
    keys = ["rmse", "mae", "acc"] + [
        k for k in model_m if k.startswith(("rmse_", "acc_"))
    ]
    for key in keys:
        model_val = model_m.get(key, float("nan"))
        persist_val = persist_m.get(key, float("nan"))

        if key.startswith("acc"):
            better = "✅" if model_val > persist_val else "❌"
        else:
            better = "✅" if model_val < persist_val else "❌"

        unit = ""
        if key.endswith("_physical"):
            unit = UNITS.get(key[len("rmse_"):-len("_physical")], "")
        print(f"  {key:<23} {model_val:>12.6f} {persist_val:>12.6f} {better:>10}  {unit}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Weather Transformer")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--output", type=str, default="results/metrics.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--figure", type=str, default=None,
        help="Also save forecast maps for one test sample to this path",
    )
    parser.add_argument(
        "--figure-sample", type=int, default=0,
        help="Test sample index for --figure",
    )
    parser.add_argument(
        "--figure-variable", type=str, default=None,
        help="Variable for --figure (default: first variable)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    variable_names = list(config["data"]["variables"])

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Load test dataset
    test_dataset = WeatherBenchDataset(data_dir=args.data_dir, split="test")
    print(f"Test samples: {len(test_dataset)}")

    # Climatology for ACC: training years, same day of year and hour as each test target
    climatology = seasonal_climatology(
        np.load(Path(args.data_dir) / "train_targets.npy", mmap_mode="r"),
        load_target_times(args.data_dir, "train"),
        load_target_times(args.data_dir, "test"),
    )

    # Evaluate (area-weighted)
    results = evaluate_model(
        model,
        test_dataset,
        device,
        variable_names=variable_names,
        lat_weights=latitude_weights(test_dataset.height),
        climatology=climatology,
    )

    stats = load_normalization_stats(args.data_dir)
    if stats:
        results = {name: add_physical_units(m, stats) for name, m in results.items()}

    results["settings"] = {
        "test_samples": len(test_dataset),
        "latitude_weighting": "cos(latitude), normalized to mean 1",
        "acc_climatology": (
            "mean of training targets at the same hour of day, within "
            f"±{CLIMATOLOGY_WINDOW_DAYS} days of the same day of year"
        ),
        "units": "normalized (z-score) unless the key ends in _physical",
        "physical_units": {v: UNITS.get(v, "") for v in variable_names},
    }

    # Print results
    print_results(results)

    # Save results. UTF-8 explicitly: the units contain "²", and on Windows the
    # default encoding (cp1252) would write a file other tools cannot read.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    if args.figure:
        save_forecast_figure(
            model,
            test_dataset,
            device,
            save_path=args.figure,
            sample_idx=args.figure_sample,
            variable_names=variable_names,
            variable=args.figure_variable or variable_names[0],
            stats=stats,
        )


if __name__ == "__main__":
    main()
