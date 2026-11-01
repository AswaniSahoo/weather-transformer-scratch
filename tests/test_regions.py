"""
Unit tests for regions and seasons.

Run with: pytest tests/test_regions.py -v
"""

import numpy as np

from src.evaluation.regions import REGIONS, months, region_cells


def test_india_box_on_the_wb2_grid():
    rows, cols = region_cells(REGIONS["india"], 32, 64)

    # Cell centres 8.4375-36.5625°N and 67.5-95.625°E
    assert rows.tolist() == [17, 18, 19, 20, 21, 22]
    assert cols.tolist() == [12, 13, 14, 15, 16, 17]


def test_months():
    times = np.array(["2020-01-31T18", "2020-06-01T00", "2020-12-31T18"], dtype="datetime64[ns]")

    assert months(times).tolist() == [1, 6, 12]
