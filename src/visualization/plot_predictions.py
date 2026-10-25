"""
Prediction visualization with cartopy world maps.

Creates side-by-side comparison plots:
    Input → Prediction → Ground Truth → Error

Uses cartopy for proper geographic projections.

Grid convention (WeatherBench2 equiangular grids): rows are latitude cell
centres ascending from the South Pole, columns are longitude cell centres
starting at 0°E and increasing eastward (0°, 5.625°, ..., 354.375° on the
64x32 grid). Fields are re-centred to [-180°, 180°] before plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not installed. Using simple plots.")


def recenter_longitudes(field: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """
    Re-centre a global field from 0..360°E onto a [-180°, 180°] map.

    Drawing a WeatherBench2 array with extent=[-180, 180] places the 0°E column
    at 180°W, shifting every feature 180° away from the coastlines. Column j is
    centred on j * 360/W °E, so the cell straddling the dateline cannot be
    drawn at [-180, 180] as a whole pixel. Each column is therefore split into
    two half-cell pixels and rolled so the first pixel starts exactly at 180°W:
    cell edges stay exact, and cartopy draws the image without regridding.

    Args:
        field: Array of shape (..., H, W), W even, first column centred on 0°E.

    Returns:
        (field, extent): the field with shape (..., H, 2W), and the extent
        [-180, 180, -90, 90] for imshow(..., origin="lower").
    """
    width = field.shape[-1]
    if width % 2:
        raise ValueError(f"Expected an even number of longitudes, got {width}")

    # Half-cell pixel p spans [(p - 1) * dlon/2, p * dlon/2] degrees east,
    # so pixel width + 1 is the first one starting at 180°E (= 180°W).
    halves = np.repeat(field, 2, axis=-1)
    halves = np.roll(halves, -(width + 1), axis=-1)
    return halves, [-180.0, 180.0, -90.0, 90.0]


def plot_prediction_comparison(
    input_data: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    variable_idx: int = 0,
    variable_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 12),
    colorbar_label: Optional[str] = None,
) -> plt.Figure:
    """
    Plot side-by-side comparison of input, prediction, ground truth, and error.

    Args:
        input_data: Input weather state (C, H, W) or (H, W).
        prediction: Model prediction (C, H, W) or (H, W).
        target: Ground truth (C, H, W) or (H, W).
        variable_idx: Which variable to plot if multi-channel.
        variable_names: List of variable names for titles.
        save_path: Path to save the figure.
        title: Overall figure title.
        figsize: Figure size.
        colorbar_label: Optional label (e.g. units) for every colorbar.

    Returns:
        matplotlib Figure object.
    """
    # Handle multi-channel inputs
    if input_data.ndim == 3:
        input_2d = input_data[variable_idx]
        pred_2d = prediction[variable_idx]
        target_2d = target[variable_idx]
    else:
        input_2d = input_data
        pred_2d = prediction
        target_2d = target

    # Compute error
    error = pred_2d - target_2d

    # Variable name for title
    if variable_names and variable_idx < len(variable_names):
        var_name = variable_names[variable_idx]
    else:
        var_name = f"Variable {variable_idx}"

    # Create figure
    if HAS_CARTOPY:
        fig = _plot_with_cartopy(
            input_2d, pred_2d, target_2d, error, var_name, figsize, colorbar_label
        )
    else:
        fig = _plot_simple(
            input_2d, pred_2d, target_2d, error, var_name, figsize, colorbar_label
        )

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved prediction plot to {save_path}")

    return fig


def _plot_with_cartopy(
    input_2d: np.ndarray,
    pred_2d: np.ndarray,
    target_2d: np.ndarray,
    error: np.ndarray,
    var_name: str,
    figsize: Tuple[int, int],
    colorbar_label: Optional[str] = None,
) -> plt.Figure:
    """Create plot with cartopy geographic projections."""
    fig, axes = plt.subplots(
        2, 2, figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    titles = [
        f"Input (t) — {var_name}",
        f"Prediction (t+6h) — {var_name}",
        f"Ground Truth (t+6h) — {var_name}",
        f"Error (Pred - Truth) — {var_name}",
    ]
    data_list = [input_2d, pred_2d, target_2d, error]

    # Common colormap limits for input/pred/target
    vmin = min(input_2d.min(), pred_2d.min(), target_2d.min())
    vmax = max(input_2d.max(), pred_2d.max(), target_2d.max())

    for idx, (ax, data, title) in enumerate(zip(axes.flat, data_list, titles)):
        data, extent = recenter_longitudes(data)

        if idx < 3:
            im = ax.imshow(
                data, origin="lower", extent=extent,
                cmap="RdBu_r", vmin=vmin, vmax=vmax,
                transform=ccrs.PlateCarree()
            )
        else:
            # Error plot with diverging colormap centered at 0
            err_max = max(abs(error.min()), abs(error.max()))
            im = ax.imshow(
                data, origin="lower", extent=extent,
                cmap="coolwarm", vmin=-err_max, vmax=err_max,
                transform=ccrs.PlateCarree()
            )

        # After imshow, which would otherwise widen the view to the image extent
        ax.set_global()
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)

        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.6, pad=0.05, label=colorbar_label)

    return fig


def _plot_simple(
    input_2d: np.ndarray,
    pred_2d: np.ndarray,
    target_2d: np.ndarray,
    error: np.ndarray,
    var_name: str,
    figsize: Tuple[int, int],
    colorbar_label: Optional[str] = None,
) -> plt.Figure:
    """Create simple plot without cartopy."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    titles = [
        f"Input (t) — {var_name}",
        f"Prediction (t+6h) — {var_name}",
        f"Ground Truth (t+6h) — {var_name}",
        f"Error (Pred - Truth) — {var_name}",
    ]
    data_list = [input_2d, pred_2d, target_2d, error]

    vmin = min(input_2d.min(), pred_2d.min(), target_2d.min())
    vmax = max(input_2d.max(), pred_2d.max(), target_2d.max())

    for idx, (ax, data, title) in enumerate(zip(axes.flat, data_list, titles)):
        data, extent = recenter_longitudes(data)

        if idx < 3:
            im = ax.imshow(data, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                           origin="lower", extent=extent)
        else:
            err_max = max(abs(error.min()), abs(error.max()))
            im = ax.imshow(data, cmap="coolwarm", vmin=-err_max, vmax=err_max,
                           origin="lower", extent=extent)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        plt.colorbar(im, ax=ax, shrink=0.8, label=colorbar_label)

    return fig


def plot_all_variables(
    input_data: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    variable_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
) -> List[plt.Figure]:
    """
    Create prediction comparison plots for all variables.
    
    Args:
        input_data: Input tensor (C, H, W).
        prediction: Prediction tensor (C, H, W).
        target: Target tensor (C, H, W).
        variable_names: List of variable names.
        save_dir: Directory to save figures.
        
    Returns:
        List of Figure objects.
    """
    n_vars = input_data.shape[0]
    if variable_names is None:
        variable_names = [f"var{i}" for i in range(n_vars)]

    figs = []
    for i in range(n_vars):
        save_path = None
        if save_dir:
            save_path = f"{save_dir}/prediction_{variable_names[i]}.png"

        fig = plot_prediction_comparison(
            input_data, prediction, target,
            variable_idx=i,
            variable_names=variable_names,
            save_path=save_path,
        )
        figs.append(fig)

    return figs


if __name__ == "__main__":
    print("Prediction Visualization — Demo")
    print("=" * 40)

    # Create synthetic data for demo
    H, W, C = 32, 64, 4
    input_data = np.random.randn(C, H, W)
    target = input_data + np.random.randn(C, H, W) * 0.3
    prediction = target + np.random.randn(C, H, W) * 0.5

    variable_names = ["t850", "z500", "u10", "v10"]

    # Synthetic noise: never save over results/figures/prediction_sample.png,
    # which is the real-data figure shown in the README.
    fig = plot_prediction_comparison(
        input_data, prediction, target,
        variable_idx=0,
        variable_names=variable_names,
        title="Weather Transformer — 6h Forecast (synthetic demo)",
    )

    print("✅ Prediction plot created!")
    plt.show()
