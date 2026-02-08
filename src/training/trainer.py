"""
Trainer class for Weather Transformer.

Handles the full training loop with:
- Physics-informed loss
- Learning rate scheduling
- Checkpointing
- Logging
- Validation
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.physics_loss import PhysicsInformedLoss


class Trainer:
    """
    Trainer for the Weather Transformer model.
    
    Handles:
    - Training loop with physics-informed loss
    - Validation
    - Checkpointing (best model + periodic)
    - Learning rate scheduling
    - Logging training metrics
    
    Args:
        model: WeatherTransformer model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Configuration dictionary.
        device: Device to train on (cuda/cpu/auto).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: Optional[str] = None,
    ):
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Model
        self.model = model.to(self.device)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Config
        self.config = config
        train_cfg = config.get("training", {})

        # Training hyperparameters
        self.epochs = train_cfg.get("epochs", 50)
        self.lr = train_cfg.get("learning_rate", 1e-4)
        self.weight_decay = train_cfg.get("weight_decay", 0.01)

        # Loss function (physics-informed)
        loss_cfg = train_cfg.get("loss", {})
        self.loss_fn = PhysicsInformedLoss(
            mse_weight=loss_cfg.get("mse_weight", 1.0),
            smoothness_weight=loss_cfg.get("smoothness_weight", 0.1),
            conservation_weight=loss_cfg.get("conservation_weight", 0.05),
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6,
        )

        # Checkpointing
        paths_cfg = config.get("paths", {})
        self.checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.checkpoint_dir / "training_log.json"

        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "val_mse": [],
            "train_smoothness": [],
            "train_conservation": [],
            "lr": [],
        }

    def train(self) -> Dict[str, List[float]]:
        """
        Run full training loop.
        
        Returns:
            Training history dictionary.
        """
        print(f"\n Starting training for {self.epochs} epochs...")
        print(f"   Checkpoints: {self.checkpoint_dir}")
        print()

        total_start = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train one epoch
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = self._validate()

            # Step scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Record history
            self.history["train_loss"].append(train_metrics["total"])
            self.history["val_loss"].append(val_metrics["total"])
            self.history["train_mse"].append(train_metrics["mse"])
            self.history["val_mse"].append(val_metrics["mse"])
            self.history["train_smoothness"].append(train_metrics["smoothness"])
            self.history["train_conservation"].append(train_metrics["conservation"])
            self.history["lr"].append(current_lr)

            # Check for best model
            is_best = val_metrics["total"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["total"]
                self.best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)

            # Periodic checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Epoch summary
            epoch_time = time.time() - epoch_start
            best_marker = " * " if is_best else ""
            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train: {train_metrics['total']:.4f} | "
                f"Val: {val_metrics['total']:.4f}{best_marker} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

        total_time = time.time() - total_start
        print(f"\n Training complete in {total_time/60:.1f} minutes")
        print(f"   Best epoch: {self.best_epoch} (val_loss = {self.best_val_loss:.6f})")

        # Save final log
        self._save_log()

        return self.history

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_smoothness = 0.0
        total_conservation = 0.0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:3d}",
            leave=False,
            ncols=100,
        )

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            predictions = self.model(inputs)

            # Loss
            losses = self.loss_fn(predictions, targets)

            # Backward
            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update
            self.optimizer.step()

            # Accumulate
            total_loss += losses["total"].item()
            total_mse += losses["mse"].item()
            total_smoothness += losses["smoothness"].item()
            total_conservation += losses["conservation"].item()
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{losses['total'].item():.4f}"})

        return {
            "total": total_loss / n_batches,
            "mse": total_mse / n_batches,
            "smoothness": total_smoothness / n_batches,
            "conservation": total_conservation / n_batches,
        }

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(inputs)
            losses = self.loss_fn(predictions, targets)

            total_loss += losses["total"].item()
            total_mse += losses["mse"].item()
            n_batches += 1

        return {
            "total": total_loss / n_batches,
            "mse": total_mse / n_batches,
        }

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)

    def _save_log(self):
        """Save training log to JSON."""
        log = {
            "config": self.config,
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }
        with open(self.log_file, "w") as f:
            json.dump(log, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]


if __name__ == "__main__":
    print("Trainer — Sanity Check")
    print("=" * 40)

    # Quick check that Trainer can be instantiated
    from src.models.weather_transformer import WeatherTransformer
    from torch.utils.data import TensorDataset, DataLoader

    # Dummy model
    model = WeatherTransformer(
        in_channels=4, out_channels=4,
        img_height=32, img_width=64,
        patch_size=4, embed_dim=64, num_heads=4, num_layers=2,
    )

    # Dummy data
    dummy_inputs = torch.randn(16, 4, 32, 64)
    dummy_targets = torch.randn(16, 4, 32, 64)
    dummy_dataset = TensorDataset(dummy_inputs, dummy_targets)
    dummy_loader = DataLoader(dummy_dataset, batch_size=4)

    # Config
    config = {
        "training": {"epochs": 2, "learning_rate": 1e-3},
        "paths": {"checkpoint_dir": "checkpoints_test"},
    }

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        config=config,
        device="cpu",
    )

    print(f"Trainer created successfully!")
    print(f"Device: {trainer.device}")
    print(f"Epochs: {trainer.epochs}")
    print(f"LR: {trainer.lr}")

    # Quick train test
    print("\nRunning quick training test (2 epochs)...")
    history = trainer.train()

    print(f"\n Trainer working!")
    print(f"   Train losses: {history['train_loss']}")
    print(f"   Val losses: {history['val_loss']}")
