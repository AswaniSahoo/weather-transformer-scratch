"""
Evaluation metrics for weather forecasting.

Implements standard weather forecast verification metrics:
1. RMSE  — Root Mean Squared Error (per variable and overall)
2. MAE   — Mean Absolute Error
3. ACC   — Anomaly Correlation Coefficient (gold standard for weather skill)

Also implements a Persistence Baseline: predict Y(t+1) = X(t).
"""

import torch
import numpy as np
from typing import Dict


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Root Mean Squared Error.
    
    Args:
        prediction: (B, C, H, W) predicted weather state.
        target: (B, C, H, W) ground truth weather state.
        
    Returns:
        Scalar RMSE averaged over all dimensions.
    """
    return torch.sqrt(((prediction - target) ** 2).mean())


def rmse_per_variable(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    RMSE computed per variable (channel).
    
    Args:
        prediction: (B, C, H, W)
        target: (B, C, H, W)
        
    Returns:
        Tensor of shape (C,) with RMSE for each variable.
    """
    # MSE per channel: average over B, H, W
    mse_per_ch = ((prediction - target) ** 2).mean(dim=(0, 2, 3))
    return torch.sqrt(mse_per_ch)


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error.
    
    Args:
        prediction: (B, C, H, W)
        target: (B, C, H, W)
        
    Returns:
        Scalar MAE.
    """
    return (prediction - target).abs().mean()


def anomaly_correlation_coefficient(
    prediction: torch.Tensor,
    target: torch.Tensor,
    climatology: torch.Tensor = None,
) -> torch.Tensor:
    """
    Anomaly Correlation Coefficient (ACC).
    
    The gold-standard metric for weather forecast skill.
    Measures the correlation between predicted and observed anomalies
    from climatology (or mean state).
    
    ACC = sum((pred - clim) * (target - clim)) /
          sqrt(sum((pred - clim)^2) * sum((target - clim)^2))
    
    Args:
        prediction: (B, C, H, W)
        target: (B, C, H, W)
        climatology: (C, H, W) or None. If None, uses mean of target.
        
    Returns:
        Scalar ACC value (higher is better, 1.0 = perfect).
    """
    if climatology is None:
        # Use the batch mean as a proxy for climatology
        climatology = target.mean(dim=0)  # (C, H, W)

    # Anomalies
    pred_anomaly = prediction - climatology.unsqueeze(0)
    target_anomaly = target - climatology.unsqueeze(0)

    # Correlation
    numerator = (pred_anomaly * target_anomaly).sum()
    denominator = torch.sqrt(
        (pred_anomaly ** 2).sum() * (target_anomaly ** 2).sum()
    )

    # Avoid division by zero
    acc = numerator / (denominator + 1e-8)

    return acc


def persistence_baseline(
    inputs: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for the persistence baseline: Y_pred = X_input.
    
    The simplest possible "forecast" — predict that nothing changes.
    Any useful model should beat this.
    
    Args:
        inputs: (B, C, H, W) input weather states.
        targets: (B, C, H, W) actual next-step weather states.
        
    Returns:
        Dict with RMSE, MAE, and ACC for the persistence baseline.
    """
    return {
        "rmse": rmse(inputs, targets).item(),
        "mae": mae(inputs, targets).item(),
        "acc": anomaly_correlation_coefficient(inputs, targets).item(),
    }


def compute_all_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    variable_names: list = None,
) -> Dict[str, float]:
    """
    Compute all metrics at once.
    
    Args:
        prediction: (B, C, H, W)
        target: (B, C, H, W)
        variable_names: Optional list of variable names for per-var metrics.
        
    Returns:
        Dict with all metric values.
    """
    metrics = {
        "rmse": rmse(prediction, target).item(),
        "mae": mae(prediction, target).item(),
        "acc": anomaly_correlation_coefficient(prediction, target).item(),
    }

    # Per-variable RMSE
    per_var_rmse = rmse_per_variable(prediction, target)
    if variable_names and len(variable_names) == per_var_rmse.shape[0]:
        for name, val in zip(variable_names, per_var_rmse):
            metrics[f"rmse_{name}"] = val.item()
    else:
        for i, val in enumerate(per_var_rmse):
            metrics[f"rmse_var{i}"] = val.item()

    return metrics
