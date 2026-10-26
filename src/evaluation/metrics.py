"""
Evaluation metrics for weather forecasting.

Implements standard weather forecast verification metrics:
1. RMSE  — Root Mean Squared Error (per variable and overall)
2. MAE   — Mean Absolute Error
3. ACC   — Anomaly Correlation Coefficient (gold standard for weather skill)

Also implements a Persistence Baseline: predict Y(t+1) = X(t).

Latitude weighting: on a lat-lon grid, cells shrink toward the poles
(area ∝ cos(latitude)), so a plain mean over the grid over-weights the polar
rows. Every metric takes optional `lat_weights` (see latitude_weights());
with them each cell counts in proportion to its area, as in WeatherBench2
(Rasp et al., 2023). Without them, every cell counts equally.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch

Dims = Optional[Union[int, Tuple[int, ...]]]


def latitude_weights(n_lat: int) -> torch.Tensor:
    """
    Area weights for the rows of an equiangular latitude grid.

    Rows are cell centres from the South Pole northward with no pole rows,
    as in WeatherBench2's equiangular grids (-87.1875°..87.1875° for 32 rows).
    For such a grid each cell's area is exactly proportional to cos(latitude).

    Args:
        n_lat: Number of latitude rows (H).

    Returns:
        Tensor of shape (H,), normalized to mean 1.
    """
    lat = -90.0 + (torch.arange(n_lat, dtype=torch.float64) + 0.5) * 180.0 / n_lat
    weights = torch.cos(torch.deg2rad(lat))
    return (weights / weights.mean()).float()


def _weighted_mean(
    x: torch.Tensor,
    lat_weights: Optional[torch.Tensor] = None,
    dim: Dims = None,
) -> torch.Tensor:
    """Mean of x (..., H, W), weighting each row (dim -2) by lat_weights if given."""
    if lat_weights is None:
        return x.mean() if dim is None else x.mean(dim=dim)

    w = lat_weights.to(device=x.device, dtype=x.dtype).view(-1, 1).expand_as(x)
    if dim is None:
        return (x * w).sum() / w.sum()
    return (x * w).sum(dim=dim) / w.sum(dim=dim)


def rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lat_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Root Mean Squared Error.

    Args:
        prediction: (B, C, H, W) predicted weather state.
        target: (B, C, H, W) ground truth weather state.
        lat_weights: Optional (H,) area weights.

    Returns:
        Scalar RMSE averaged over all dimensions.
    """
    return torch.sqrt(_weighted_mean((prediction - target) ** 2, lat_weights))


def rmse_per_variable(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lat_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    RMSE computed per variable (channel).

    Args:
        prediction: (B, C, H, W)
        target: (B, C, H, W)
        lat_weights: Optional (H,) area weights.

    Returns:
        Tensor of shape (C,) with RMSE for each variable.
    """
    # MSE per channel: average over B, H, W
    mse_per_ch = _weighted_mean((prediction - target) ** 2, lat_weights, dim=(0, 2, 3))
    return torch.sqrt(mse_per_ch)


def mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lat_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Mean Absolute Error.

    Args:
        prediction: (B, C, H, W)
        target: (B, C, H, W)
        lat_weights: Optional (H,) area weights.

    Returns:
        Scalar MAE.
    """
    return _weighted_mean((prediction - target).abs(), lat_weights)


def acc_per_variable(
    prediction: torch.Tensor,
    target: torch.Tensor,
    climatology: Optional[torch.Tensor] = None,
    lat_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Anomaly Correlation Coefficient per variable (WeatherBench2 definition).

    For every sample and variable, correlates the forecast and observed
    anomalies from climatology over the globe:

        ACC = sum(w * f' * o') / sqrt(sum(w * f'^2) * sum(w * o'^2)),
        f' = prediction - climatology,  o' = target - climatology

    then averages over samples. Computing it per variable keeps fields with
    different units/variability from being mixed into one correlation.

    Args:
        prediction: (B, C, H, W)
        target: (B, C, H, W)
        climatology: (C, H, W) or None. Should come from the training period;
            if None, the mean of `target` is used as a rough proxy (which uses
            information from the evaluation period itself).
        lat_weights: Optional (H,) area weights.

    Returns:
        Tensor of shape (C,) with ACC for each variable (1.0 = perfect).
    """
    if climatology is None:
        # Use the batch mean as a proxy for climatology
        climatology = target.mean(dim=0)  # (C, H, W)

    # Anomalies
    pred_anomaly = prediction - climatology
    target_anomaly = target - climatology

    # Spatial correlation per (sample, variable): (B, C)
    spatial = (-2, -1)
    numerator = _weighted_mean(pred_anomaly * target_anomaly, lat_weights, dim=spatial)
    denominator = torch.sqrt(
        _weighted_mean(pred_anomaly ** 2, lat_weights, dim=spatial)
        * _weighted_mean(target_anomaly ** 2, lat_weights, dim=spatial)
    )

    # Avoid division by zero
    acc = numerator / (denominator + 1e-8)

    return acc.mean(dim=0)


def anomaly_correlation_coefficient(
    prediction: torch.Tensor,
    target: torch.Tensor,
    climatology: Optional[torch.Tensor] = None,
    lat_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Anomaly Correlation Coefficient (ACC), averaged over variables.

    The gold-standard metric for weather forecast skill. See
    acc_per_variable() for the definition.

    Args:
        prediction: (B, C, H, W)
        target: (B, C, H, W)
        climatology: (C, H, W) or None (see acc_per_variable).
        lat_weights: Optional (H,) area weights.

    Returns:
        Scalar ACC value (higher is better, 1.0 = perfect).
    """
    return acc_per_variable(prediction, target, climatology, lat_weights).mean()


def persistence_baseline(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    variable_names: Optional[List[str]] = None,
    lat_weights: Optional[torch.Tensor] = None,
    climatology: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute metrics for the persistence baseline: Y_pred = X_input.

    The simplest possible "forecast" — predict that nothing changes.
    Any useful model should beat this.

    Args:
        inputs: (B, C, H, W) input weather states.
        targets: (B, C, H, W) actual next-step weather states.
        variable_names: Optional list of variable names for per-var metrics.
        lat_weights: Optional (H,) area weights.
        climatology: Optional (C, H, W) climatology for ACC.

    Returns:
        Dict with the same keys as compute_all_metrics().
    """
    return compute_all_metrics(inputs, targets, variable_names, lat_weights, climatology)


def compute_all_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    variable_names: Optional[List[str]] = None,
    lat_weights: Optional[torch.Tensor] = None,
    climatology: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute all metrics at once.

    Args:
        prediction: (B, C, H, W)
        target: (B, C, H, W)
        variable_names: Optional list of variable names for per-var metrics.
        lat_weights: Optional (H,) area weights.
        climatology: Optional (C, H, W) climatology for ACC.

    Returns:
        Dict with overall rmse/mae/acc plus rmse_<var> and acc_<var>.
    """
    per_var_acc = acc_per_variable(prediction, target, climatology, lat_weights)

    metrics = {
        "rmse": rmse(prediction, target, lat_weights).item(),
        "mae": mae(prediction, target, lat_weights).item(),
        "acc": per_var_acc.mean().item(),
    }

    # Per-variable RMSE and ACC
    per_var_rmse = rmse_per_variable(prediction, target, lat_weights)
    if variable_names and len(variable_names) == per_var_rmse.shape[0]:
        names = list(variable_names)
    else:
        names = [f"var{i}" for i in range(per_var_rmse.shape[0])]

    for name, rmse_val, acc_val in zip(names, per_var_rmse, per_var_acc):
        metrics[f"rmse_{name}"] = rmse_val.item()
        metrics[f"acc_{name}"] = acc_val.item()

    return metrics
