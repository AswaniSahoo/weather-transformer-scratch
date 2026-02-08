"""
Evaluation script for the Weather Transformer.

Loads a trained model, runs it on the test set, and compares
against the persistence baseline.

Usage:
    python -m src.evaluation.evaluate
    python -m src.evaluation.evaluate --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import yaml

from src.data.dataset import WeatherBenchDataset
from src.models.weather_transformer import WeatherTransformer
from src.evaluation.metrics import (
    compute_all_metrics,
    persistence_baseline,
)


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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")

    return model


@torch.no_grad()
def evaluate_model(
    model: WeatherTransformer,
    dataset: WeatherBenchDataset,
    device: str,
    batch_size: int = 64,
    variable_names: list = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on a dataset and compare with persistence baseline.
    
    Args:
        model: Trained WeatherTransformer.
        dataset: Test dataset.
        device: Device string.
        batch_size: Batch size for evaluation.
        variable_names: List of variable names.
        
    Returns:
        Dict with 'model' and 'persistence' metric dicts.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    all_preds = []
    all_targets = []
    all_inputs = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)

        all_preds.append(predictions.cpu())
        all_targets.append(targets.cpu())
        all_inputs.append(inputs.cpu())

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_inputs = torch.cat(all_inputs, dim=0)

    # Model metrics
    model_metrics = compute_all_metrics(all_preds, all_targets, variable_names)

    # Persistence baseline metrics
    persist_metrics = persistence_baseline(all_inputs, all_targets)

    return {
        "model": model_metrics,
        "persistence": persist_metrics,
    }


def print_results(results: Dict[str, Dict[str, float]]):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Model':>12} {'Persistence':>12} {'Better?':>10}")
    print("-" * 60)

    model_m = results["model"]
    persist_m = results["persistence"]

    for key in ["rmse", "mae", "acc"]:
        model_val = model_m.get(key, float("nan"))
        persist_val = persist_m.get(key, float("nan"))

        if key == "acc":
            better = "✅" if model_val > persist_val else "❌"
        else:
            better = "✅" if model_val < persist_val else "❌"

        print(f"  {key.upper():<23} {model_val:>12.6f} {persist_val:>12.6f} {better:>10}")

    # Per-variable RMSE
    var_keys = [k for k in model_m if k.startswith("rmse_")]
    if var_keys:
        print(f"\nPer-Variable RMSE:")
        for key in var_keys:
            print(f"  {key:<23} {model_m[key]:>12.6f}")

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
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Load test dataset
    test_dataset = WeatherBenchDataset(data_dir=args.data_dir, split="test")
    print(f"Test samples: {len(test_dataset)}")

    # Evaluate
    results = evaluate_model(model, test_dataset, device)

    # Print results
    print_results(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
