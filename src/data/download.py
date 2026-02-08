"""
Download WeatherBench2 ERA5 sample data from Google Cloud Storage.

Downloads a subset of ERA5 reanalysis data (2015-2020) at 6-hourly resolution
on a 64x32 equiangular grid. Variables: temperature (850hPa), geopotential (500hPa),
u-wind (10m), v-wind (10m).

Usage:
    python -m src.data.download
    python -m src.data.download --start-year 2015 --end-year 2020
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import xarray as xr
import yaml


# WeatherBench2 GCS path for ERA5 data
GCS_PATH = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

# Default variables to download
DEFAULT_VARIABLES = {
    "2m_temperature": None,           # Surface temperature (no level)
    "geopotential": 500,              # z500 - 500 hPa geopotential
    "u_component_of_wind": 10,        # u10 - 10m u-wind (as level selector)
    "v_component_of_wind": 10,        # v10 - 10m v-wind (as level selector)
}

# Alternative: pressure level variables
PRESSURE_LEVEL_VARIABLES = {
    "temperature": 850,               # t850 - 850 hPa temperature
    "geopotential": 500,              # z500 - 500 hPa geopotential
}

SURFACE_VARIABLES = [
    "10m_u_component_of_wind",        # u10
    "10m_v_component_of_wind",        # v10
]


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_weatherbench2(
    output_dir: str = "data/raw",
    start_year: int = 2015,
    end_year: int = 2020,
    variables: Optional[List[str]] = None,
    config_path: Optional[str] = None,
) -> xr.Dataset:
    """
    Download WeatherBench2 ERA5 data from Google Cloud Storage.
    
    Args:
        output_dir: Directory to save the downloaded data
        start_year: First year to download (inclusive)
        end_year: Last year to download (inclusive)
        variables: List of variable names to download. If None, uses defaults.
        config_path: Path to config file. If provided, overrides other args.
    
    Returns:
        xr.Dataset: The downloaded dataset
    
    Notes:
        - Data is downloaded from the WeatherBench2 bucket on GCS
        - The 64x32 grid is ~5.625° resolution (coarse but fast to train on)
        - 6-hourly temporal resolution (00:00, 06:00, 12:00, 18:00 UTC)
    """
    # Load config if provided
    if config_path:
        config = load_config(config_path)
        data_config = config.get("data", {})
        start_year = data_config.get("train_years", [2015, 2018])[0]
        end_year = data_config.get("test_years", [2020, 2020])[1]
        output_dir = data_config.get("data_dir", output_dir)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Opening WeatherBench2 ERA5 dataset from GCS...")
    print(f"Source: {GCS_PATH}")
    
    # Open the zarr dataset from GCS
    # This uses gcsfs under the hood for anonymous access
    import gcsfs
    fs = gcsfs.GCSFileSystem(token='anon')
    store = fs.get_mapper(GCS_PATH)
    # consolidated=False since .zmetadata may not exist in all datasets
    ds = xr.open_zarr(store, consolidated=False)
    
    print(f"Full dataset shape: {dict(ds.dims)}")
    print(f"Available variables: {list(ds.data_vars)}")
    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    # Select time range
    time_slice = slice(f"{start_year}-01-01", f"{end_year}-12-31")
    ds_subset = ds.sel(time=time_slice)
    
    print(f"\nSelected time range: {start_year}-{end_year}")
    print(f"Subset shape: {dict(ds_subset.dims)}")
    
    # Select variables
    # For WeatherBench2, we need to handle both single-level and pressure-level variables
    selected_vars = []
    
    # Check available variables and select appropriately
    available_vars = list(ds_subset.data_vars)
    
    # Try to get temperature at 850 hPa
    if "temperature" in available_vars:
        if "level" in ds_subset["temperature"].dims:
            temp = ds_subset["temperature"].sel(level=850, drop=True)
            temp = temp.rename("t850")
            selected_vars.append(temp)
            print(f"Selected: temperature at 850 hPa -> t850")
    
    # Try to get geopotential at 500 hPa
    if "geopotential" in available_vars:
        if "level" in ds_subset["geopotential"].dims:
            geo = ds_subset["geopotential"].sel(level=500, drop=True)
            geo = geo.rename("z500")
            selected_vars.append(geo)
            print(f"Selected: geopotential at 500 hPa -> z500")
    
    # Try to get 10m winds
    for var_name, short_name in [
        ("10m_u_component_of_wind", "u10"),
        ("10m_v_component_of_wind", "v10"),
    ]:
        if var_name in available_vars:
            wind = ds_subset[var_name].rename(short_name)
            selected_vars.append(wind)
            print(f"Selected: {var_name} -> {short_name}")
    
    # Fallback: if standard names not found, try alternative names
    if not selected_vars:
        print("\nStandard variable names not found. Trying alternatives...")
        # Try 2m_temperature as fallback for temperature
        if "2m_temperature" in available_vars:
            temp = ds_subset["2m_temperature"].rename("t2m")
            selected_vars.append(temp)
            print(f"Selected: 2m_temperature -> t2m")
    
    if not selected_vars:
        raise ValueError(
            f"Could not find expected variables. Available: {available_vars}"
        )
    
    # Merge selected variables into a single dataset
    # Use compat='override' to handle any coordinate mismatches
    ds_final = xr.merge(selected_vars, compat='override')
    
    print(f"\nFinal dataset variables: {list(ds_final.data_vars)}")
    print(f"Final dataset shape: {dict(ds_final.dims)}")
    
    # Compute the data (load into memory) for the subset
    # This is important for smaller time ranges to avoid repeated GCS calls
    print(f"\nDownloading data to memory...")
    ds_final = ds_final.compute()
    
    # Save to NetCDF
    output_file = output_path / f"era5_{start_year}_{end_year}.nc"
    print(f"\nSaving to {output_file}...")
    ds_final.to_netcdf(output_file)
    
    print(f"Download complete! File size: {output_file.stat().st_size / 1e6:.1f} MB")
    
    return ds_final


def check_data_exists(output_dir: str = "data/raw", start_year: int = 2015, end_year: int = 2020) -> bool:
    """Check if data file already exists."""
    output_file = Path(output_dir) / f"era5_{start_year}_{end_year}.nc"
    return output_file.exists()


def main():
    """Main entry point for data download."""
    parser = argparse.ArgumentParser(
        description="Download WeatherBench2 ERA5 data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (overrides other arguments)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save downloaded data",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="First year to download (inclusive)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2020,
        help="Last year to download (inclusive)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    
    args = parser.parse_args()
    
    # Check if already downloaded
    if not args.force and check_data_exists(args.output_dir, args.start_year, args.end_year):
        print(f"Data already exists at {args.output_dir}/era5_{args.start_year}_{args.end_year}.nc")
        print("Use --force to re-download.")
        return
    
    # Download data
    download_weatherbench2(
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
