"""
Download WeatherBench2 ERA5 sample data from Google Cloud Storage.

Downloads a subset of ERA5 reanalysis data (2015-2020) at 6-hourly resolution
on a 64x32 equiangular grid. The source path, variables, and pressure levels
come from `data` in configs/default.yaml:
    t850 — temperature at 850 hPa
    z500 — geopotential at 500 hPa
    u10  — 10 m u-wind
    v10  — 10 m v-wind

Usage:
    python -m src.data.download
    python -m src.data.download --config configs/default.yaml --start-year 2015 --end-year 2020
"""

import argparse
from pathlib import Path
from typing import Dict

import xarray as xr
import yaml


# WeatherBench2 GCS path for ERA5 data
GCS_PATH = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def select_variables(ds: xr.Dataset, variables: Dict[str, dict]) -> xr.Dataset:
    """
    Pick the model channels out of a WeatherBench2 dataset.

    Args:
        ds: WeatherBench2 dataset (lazily opened is fine).
        variables: Ordered mapping of short name -> {"name": WeatherBench2
            variable, "level": pressure level in hPa (omit for single-level
            variables)}, e.g. {"t850": {"name": "temperature", "level": 850}}.

    Returns:
        Dataset with one variable per short name, in the given order
        (this order becomes the model's channel order).

    Raises:
        KeyError: If a variable or pressure level is missing, instead of
            silently producing a different set of channels.
        ValueError: If a pressure-level variable has no level selected.
    """
    selected = {}
    for short_name, spec in variables.items():
        name = spec["name"]
        if name not in ds.data_vars:
            raise KeyError(f"{name!r} not found. Available: {sorted(ds.data_vars)}")

        data = ds[name]
        level = spec.get("level")
        if level is not None:
            data = data.sel(level=level, drop=True)
            print(f"Selected: {name} at {level} hPa -> {short_name}")
        elif "level" in data.dims:
            raise ValueError(f"{name!r} has pressure levels; set 'level' for {short_name!r}")
        else:
            print(f"Selected: {name} -> {short_name}")

        selected[short_name] = data

    return xr.Dataset(selected)


def download_weatherbench2(
    variables: Dict[str, dict],
    output_dir: str = "data/raw",
    start_year: int = 2015,
    end_year: int = 2020,
    source: str = GCS_PATH,
) -> xr.Dataset:
    """
    Download WeatherBench2 ERA5 data from Google Cloud Storage.

    Args:
        variables: Channels to download (see select_variables).
        output_dir: Directory to save the downloaded data
        start_year: First year to download (inclusive)
        end_year: Last year to download (inclusive)
        source: WeatherBench2 Zarr store to read from.

    Returns:
        xr.Dataset: The downloaded dataset

    Notes:
        - Data is downloaded from the WeatherBench2 bucket on GCS
        - The 64x32 grid is ~5.625° resolution (coarse but fast to train on)
        - 6-hourly temporal resolution (00:00, 06:00, 12:00, 18:00 UTC)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Opening WeatherBench2 ERA5 dataset from GCS...")
    print(f"Source: {source}")

    # Open the zarr dataset from GCS
    # This uses gcsfs under the hood for anonymous access
    import gcsfs
    fs = gcsfs.GCSFileSystem(token='anon')
    store = fs.get_mapper(source)
    ds = xr.open_zarr(store)

    print(f"Full dataset shape: {dict(ds.sizes)}")
    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")

    # Select time range
    time_slice = slice(f"{start_year}-01-01", f"{end_year}-12-31")
    ds_subset = ds.sel(time=time_slice)

    print(f"\nSelected time range: {start_year}-{end_year}")
    print(f"Subset shape: {dict(ds_subset.sizes)}")

    # Select variables (single-level and pressure-level)
    ds_final = select_variables(ds_subset, variables)

    print(f"\nFinal dataset variables: {list(ds_final.data_vars)}")
    print(f"Final dataset shape: {dict(ds_final.sizes)}")

    # Compute the data (load into memory) for the subset
    # This is important for smaller time ranges to avoid repeated GCS calls
    print("\nDownloading data to memory...")
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
        default="configs/default.yaml",
        help="Config file with the data source, variables, and years",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save downloaded data (default: paths.data_dir)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="First year to download, inclusive (default: first training year)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last year to download, inclusive (default: last test year)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )

    args = parser.parse_args()

    # Command-line arguments override the config
    config = load_config(args.config)
    data_config = config["data"]
    output_dir = args.output_dir or config.get("paths", {}).get("data_dir", "data/raw")
    start_year = args.start_year or data_config["train_years"][0]
    end_year = args.end_year or data_config["test_years"][1]

    # Check if already downloaded
    if not args.force and check_data_exists(output_dir, start_year, end_year):
        print(f"Data already exists at {output_dir}/era5_{start_year}_{end_year}.nc")
        print("Use --force to re-download.")
        return

    # Download data
    download_weatherbench2(
        variables=data_config["variables"],
        output_dir=output_dir,
        start_year=start_year,
        end_year=end_year,
        source=data_config.get("source", GCS_PATH),
    )


if __name__ == "__main__":
    main()
