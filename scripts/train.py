"""
Main training entry point for the Weather Transformer.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --epochs 10
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.weather_transformer import WeatherTransformer
from src.data.dataset import WeatherBenchDataset, create_dataloaders
from src.training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Weather Transformer")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override processed data directory",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.data_dir:
        config["paths"]["processed_dir"] = args.data_dir

    print("🌦️  Weather Transformer — Training")
    print("=" * 50)

    # Data
    data_dir = config["paths"].get("processed_dir", "data/processed")
    batch_size = config["training"].get("batch_size", 32)

    print(f"Loading data from {data_dir}...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
    )
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val:   {len(val_loader.dataset)} samples")

    # Model
    model_cfg = config.get("model", {})
    model = WeatherTransformer(
        in_channels=model_cfg.get("in_channels", 4),
        out_channels=model_cfg.get("out_channels", 4),
        img_height=config["data"].get("lat_size", 32),
        img_width=config["data"].get("lon_size", 64),
        patch_size=model_cfg.get("patch_size", 4),
        embed_dim=model_cfg.get("embed_dim", 256),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_layers", 6),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        dropout=model_cfg.get("dropout", 0.1),
    )

    print(f"Model parameters: {model.count_parameters():,}")

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
    )

    # Train
    history = trainer.train()

    print("\n✅ Training finished!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    print(f"Training log saved to: {trainer.log_file}")


if __name__ == "__main__":
    main()
