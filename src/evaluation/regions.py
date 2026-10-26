"""
Regions and seasons for regional evaluation.

A region is a latitude/longitude box and covers the grid cells whose centres
fall inside it. On the 5.625° grid the India box is 6 x 6 cells (centres
8.4-36.6°N, 67.5-95.6°E), so it also covers parts of neighbouring countries,
the Arabian Sea and the Bay of Bengal.
"""

from typing import Tuple

import numpy as np

# name -> (lat_min, lat_max, lon_min, lon_max) in degrees north and east (0-360)
REGIONS = {
    "india": (6.0, 38.0, 66.0, 98.0),
}

# name -> months; June-September is India's southwest (summer) monsoon
SEASONS = {
    "monsoon_jjas": (6, 7, 8, 9),
    "rest_of_year": (10, 11, 12, 1, 2, 3, 4, 5),
}


def grid_centres(n_lat: int, n_lon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cell centres of the WeatherBench2 equiangular grid (see check_grid)."""
    lat = -90.0 + (np.arange(n_lat) + 0.5) * 180.0 / n_lat
    lon = np.arange(n_lon) * 360.0 / n_lon
    return lat, lon


def region_cells(
    box: Tuple[float, float, float, float], n_lat: int, n_lon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Row and column indices of the cells whose centres lie inside `box`."""
    lat_min, lat_max, lon_min, lon_max = box
    lat, lon = grid_centres(n_lat, n_lon)
    rows = np.flatnonzero((lat >= lat_min) & (lat <= lat_max))
    cols = np.flatnonzero((lon >= lon_min) & (lon <= lon_max))
    return rows, cols


def months(times: np.ndarray) -> np.ndarray:
    """Month (1-12) of each datetime64 value."""
    return times.astype("datetime64[M]").astype(int) % 12 + 1
