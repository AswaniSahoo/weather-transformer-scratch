"""
Training loss curve visualization.

Creates plots for:
- Training and validation loss curves
- Per-component loss breakdown (MSE, smoothness, conservation)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        history: Training history dict with keys like:
            - train_loss, val_loss
            - train_mse, val_mse
            - train_smoothness, train_conservation
            - lr
        save_path: Path to save the figure.
        figsize: Figure size.
        
    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Training Curves — Weather Transformer", fontsize=14, fontweight="bold")

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # --- Plot 1: Total Loss ---
    ax1 = axes[0, 0]
    if "train_loss" in history:
        ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.set_title("Total Loss (Physics-Informed)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: MSE Component ---
    ax2 = axes[0, 1]
    if "train_mse" in history:
        ax2.plot(epochs, history["train_mse"], "b-", label="Train MSE", linewidth=2)
    if "val_mse" in history:
        ax2.plot(epochs, history["val_mse"], "r-", label="Val MSE", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.set_title("MSE Loss Component")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Physics Loss Components ---
    ax3 = axes[1, 0]
    if "train_smoothness" in history:
        ax3.plot(epochs, history["train_smoothness"], "g-", label="Smoothness", linewidth=2)
    if "train_conservation" in history:
        ax3.plot(epochs, history["train_conservation"], "m-", label="Conservation", linewidth=2)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss Value")
    ax3.set_title("Physics Regularization Terms")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Learning Rate ---
    ax4 = axes[1, 1]
    if "lr" in history:
        ax4.plot(epochs, history["lr"], "k-", linewidth=2)
        ax4.set_yscale("log")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Learning Rate")
    ax4.set_title("Learning Rate Schedule (Cosine)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved loss curves to {save_path}")

    return fig


def plot_loss_comparison(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Compare loss curves from multiple training runs.
    
    Args:
        histories: Dict mapping run names to history dicts.
        save_path: Path to save the figure.
        figsize: Figure size.
        
    Returns:
        matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for (name, history), color in zip(histories.items(), colors):
        epochs = range(1, len(history.get("train_loss", [])) + 1)

        if "train_loss" in history:
            ax1.plot(epochs, history["train_loss"], color=color, 
                    linestyle="-", label=f"{name} (train)", linewidth=2)
        if "val_loss" in history:
            ax1.plot(epochs, history["val_loss"], color=color,
                    linestyle="--", label=f"{name} (val)", linewidth=2)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.set_title("Training Comparison")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Final values bar chart
    final_losses = {}
    for name, history in histories.items():
        if "val_loss" in history and history["val_loss"]:
            final_losses[name] = history["val_loss"][-1]

    if final_losses:
        names = list(final_losses.keys())
        losses = list(final_losses.values())
        ax2.bar(names, losses, color=colors[:len(names)])
        ax2.set_ylabel("Final Val Loss")
        ax2.set_title("Final Validation Loss")
        ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison to {save_path}")

    return fig


def load_training_log(log_path: str) -> Dict:
    """Load training log from JSON file."""
    with open(log_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    print("Loss Curve Visualization — Demo")
    print("=" * 40)

    # Create synthetic training history for demo
    n_epochs = 50
    t = np.linspace(0, 1, n_epochs)

    history = {
        "train_loss": (1.5 * np.exp(-3 * t) + 0.3 + np.random.randn(n_epochs) * 0.02).tolist(),
        "val_loss": (1.5 * np.exp(-3 * t) + 0.35 + np.random.randn(n_epochs) * 0.03).tolist(),
        "train_mse": (1.2 * np.exp(-3 * t) + 0.25 + np.random.randn(n_epochs) * 0.02).tolist(),
        "val_mse": (1.2 * np.exp(-3 * t) + 0.28 + np.random.randn(n_epochs) * 0.02).tolist(),
        "train_smoothness": (0.3 * np.exp(-2 * t) + 0.05).tolist(),
        "train_conservation": (0.1 * np.exp(-2 * t) + 0.02).tolist(),
        "lr": (1e-4 * (1 + np.cos(np.pi * t)) / 2 + 1e-6).tolist(),
    }

    fig = plot_training_curves(
        history,
        save_path="results/figures/loss_curves.png",
    )

    print("✅ Loss curves created!")
    plt.show()
