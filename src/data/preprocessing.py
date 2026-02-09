"""
Preprocessing pipeline for WeatherBench2 ERA5 data.

Handles:
- Loading raw NetCDF data
- Computing normalization statistics (per-variable mean/std)
- Creating input-target pairs: X(t) -> Y(t+1) for 6-hour forecasting
- Train/Val/Test splits by year
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import yaml


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(
    data_path: str = "data/raw/era5_2015_2020.nc",
) -> xr.Dataset:
    """
    Load raw NetCDF data downloaded from WeatherBench2.
    
    Args:
        data_path: Path to the NetCDF file
        
    Returns:
        xr.Dataset containing the weather variables
    """
    print(f"Loading raw data from {data_path}...")
    ds = xr.open_dataset(data_path)
    print(f"Loaded dataset with variables: {list(ds.data_vars)}")
    print(f"Shape: {dict(ds.dims)}")
    return ds


def compute_normalization_stats(
    ds: xr.Dataset,
    variables: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-variable mean and standard deviation for normalization.
    
    Args:
        ds: xarray Dataset with weather variables
        variables: List of variable names. If None, uses all data_vars.
        
    Returns:
        Dict mapping variable name to {"mean": float, "std": float}
    """
    if variables is None:
        variables = list(ds.data_vars)
    
    stats = {}
    for var in variables:
        if var in ds.data_vars:
            data = ds[var].values
            mean_val = float(np.nanmean(data))
            std_val = float(np.nanstd(data))
            # Prevent division by zero
            if std_val < 1e-6:
                std_val = 1.0
            stats[var] = {"mean": mean_val, "std": std_val}
            print(f"  {var}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    return stats


def normalize_data(
    ds: xr.Dataset,
    stats: Dict[str, Dict[str, float]],
) -> xr.Dataset:
    """
    Normalize dataset using precomputed statistics.
    
    Args:
        ds: xarray Dataset to normalize
        stats: Dict from compute_normalization_stats()
        
    Returns:
        Normalized xarray Dataset
    """
    ds_norm = ds.copy()
    for var, var_stats in stats.items():
        if var in ds_norm.data_vars:
            ds_norm[var] = (ds_norm[var] - var_stats["mean"]) / var_stats["std"]
    return ds_norm


def denormalize_data(
    ds: xr.Dataset,
    stats: Dict[str, Dict[str, float]],
) -> xr.Dataset:
    """
    Denormalize dataset (inverse of normalize_data).
    
    Args:
        ds: Normalized xarray Dataset
        stats: Dict from compute_normalization_stats()
        
    Returns:
        Denormalized xarray Dataset
    """
    ds_denorm = ds.copy()
    for var, var_stats in stats.items():
        if var in ds_denorm.data_vars:
            ds_denorm[var] = ds_denorm[var] * var_stats["std"] + var_stats["mean"]
    return ds_denorm


def split_by_year(
    ds: xr.Dataset,
    train_years: Tuple[int, int] = (2015, 2018),
    val_years: Tuple[int, int] = (2019, 2019),
    test_years: Tuple[int, int] = (2020, 2020),
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Split dataset into train/val/test by year.
    
    Args:
        ds: xarray Dataset with time dimension
        train_years: (start_year, end_year) inclusive
        val_years: (start_year, end_year) inclusive
        test_years: (start_year, end_year) inclusive
        
    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    def select_years(dataset: xr.Dataset, start: int, end: int) -> xr.Dataset:
        return dataset.sel(time=slice(f"{start}-01-01", f"{end}-12-31"))
    
    train_ds = select_years(ds, train_years[0], train_years[1])
    val_ds = select_years(ds, val_years[0], val_years[1])
    test_ds = select_years(ds, test_years[0], test_years[1])
    
    print(f"Train: {train_years[0]}-{train_years[1]}, {len(train_ds.time)} timesteps")
    print(f"Val:   {val_years[0]}-{val_years[1]}, {len(val_ds.time)} timesteps")
    print(f"Test:  {test_years[0]}-{test_years[1]}, {len(test_ds.time)} timesteps")
    
    return train_ds, val_ds, test_ds


def create_input_target_pairs(
    ds: xr.Dataset,
    lead_time_steps: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-target pairs for forecasting.
    
    For each timestep t, creates:
        Input: X(t)
        Target: Y(t + lead_time_steps)
    
    Args:
        ds: xarray Dataset with shape (time, lat, lon) per variable
        lead_time_steps: Number of timesteps ahead to predict (1 = 6 hours)
        
    Returns:
        Tuple of (inputs, targets) as numpy arrays
        Shape: (N_samples, C, H, W) where C = number of variables
    """
    variables = list(ds.data_vars)
    n_vars = len(variables)
    n_times = len(ds.time)
    n_lat = len(ds.latitude) if "latitude" in ds.dims else len(ds.lat)
    n_lon = len(ds.longitude) if "longitude" in ds.dims else len(ds.lon)
    
    # Number of valid samples (we lose lead_time_steps at the end)
    n_samples = n_times - lead_time_steps
    
    # Stack variables into single array: (time, channels, ...)
    # xarray data has shape (time, lon, lat) so we need to transpose to (time, channels, lat, lon)
    # Model expects (N, C, H, W) = (N, C, lat, lon) = (N, C, 32, 64)
    data_stack = np.stack(
        [ds[var].values for var in variables],
        axis=1  # Stack along channel dimension
    )
    # Transpose from (time, channels, lon, lat) to (time, channels, lat, lon)
    # i.e., swap the last two dimensions
    data_stack = np.transpose(data_stack, (0, 1, 3, 2))
    
    # Create input-target pairs
    inputs = data_stack[:n_samples]  # X(t) for t = 0, 1, ..., n_samples-1
    targets = data_stack[lead_time_steps:]  # Y(t+lead) for same t values
    
    print(f"Created {n_samples} input-target pairs")
    print(f"Input shape:  {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    
    return inputs, targets


def preprocess_and_save(
    raw_data_path: str = "data/raw/era5_2015_2020.nc",
    output_dir: str = "data/processed",
    config_path: str = "configs/default.yaml",
) -> Dict[str, str]:
    """
    Full preprocessing pipeline: load, normalize, split, create pairs, save.
    
    Args:
        raw_data_path: Path to raw NetCDF file
        output_dir: Directory to save processed data
        config_path: Path to config file
        
    Returns:
        Dict with paths to saved files
    """
    # Load config
    config = load_config(config_path)
    data_config = config.get("data", {})
    
    train_years = tuple(data_config.get("train_years", [2015, 2018]))
    val_years = tuple(data_config.get("val_years", [2019, 2019]))
    test_years = tuple(data_config.get("test_years", [2020, 2020]))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    print("\n" + "="*60)
    print("Step 1: Loading raw data")
    print("="*60)
    ds = load_raw_data(raw_data_path)
    
    # Compute normalization statistics on training data only
    print("\n" + "="*60)
    print("Step 2: Computing normalization statistics (from training data)")
    print("="*60)
    train_ds_raw, _, _ = split_by_year(ds, train_years, val_years, test_years)
    norm_stats = compute_normalization_stats(train_ds_raw)
    
    # Save normalization stats
    stats_path = output_path / "normalization_stats.json"
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved normalization stats to {stats_path}")
    
    # Normalize full dataset using training stats
    print("\n" + "="*60)
    print("Step 3: Normalizing data")
    print("="*60)
    ds_norm = normalize_data(ds, norm_stats)
    
    # Split normalized data
    print("\n" + "="*60)
    print("Step 4: Splitting data by year")
    print("="*60)
    train_ds, val_ds, test_ds = split_by_year(
        ds_norm, train_years, val_years, test_years
    )
    
    # Create input-target pairs for each split
    print("\n" + "="*60)
    print("Step 5: Creating input-target pairs")
    print("="*60)
    
    saved_paths = {"stats": str(stats_path)}
    
    for split_name, split_ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        print(f"\n--- {split_name} split ---")
        inputs, targets = create_input_target_pairs(split_ds, lead_time_steps=1)
        
        # Save as .npy files
        inputs_path = output_path / f"{split_name}_inputs.npy"
        targets_path = output_path / f"{split_name}_targets.npy"
        
        np.save(inputs_path, inputs.astype(np.float32))
        np.save(targets_path, targets.astype(np.float32))
        
        print(f"Saved {split_name} inputs to {inputs_path}")
        print(f"Saved {split_name} targets to {targets_path}")
        
        saved_paths[f"{split_name}_inputs"] = str(inputs_path)
        saved_paths[f"{split_name}_targets"] = str(targets_path)
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    
    return saved_paths


def load_processed_data(
    processed_dir: str = "data/processed",
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Load preprocessed data and normalization stats.
    
    Args:
        processed_dir: Directory with processed .npy files
        split: One of "train", "val", "test"
        
    Returns:
        Tuple of (inputs, targets, normalization_stats)
    """
    processed_path = Path(processed_dir)
    
    inputs = np.load(processed_path / f"{split}_inputs.npy")
    targets = np.load(processed_path / f"{split}_targets.npy")
    
    with open(processed_path / "normalization_stats.json", "r") as f:
        stats = json.load(f)
    
    return inputs, targets, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess WeatherBench2 data")
    parser.add_argument(
        "--raw-data",
        type=str,
        default="data/raw/era5_2015_2020.nc",
        help="Path to raw NetCDF file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    
    args = parser.parse_args()
    
    preprocess_and_save(
        raw_data_path=args.raw_data,
        output_dir=args.output_dir,
        config_path=args.config,
    )
