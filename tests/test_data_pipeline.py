"""
Unit tests for the download/preprocessing logic, on synthetic WeatherBench2-like data.

Run with: pytest tests/test_data_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from src.data.download import select_variables  # noqa: E402
from src.data.preprocessing import (  # noqa: E402
    check_grid,
    create_input_target_pairs,
    lead_time_to_steps,
    target_times,
)

N_TIME, N_LON, N_LAT = 5, 8, 4
VARIABLES = {
    "t850": {"name": "temperature", "level": 850},
    "z500": {"name": "geopotential", "level": 500},
    "u10": {"name": "10m_u_component_of_wind"},
    "v10": {"name": "10m_v_component_of_wind"},
}


def wb2_like_dataset() -> xr.Dataset:
    """Synthetic dataset with WeatherBench2's names, dim order, and grid."""
    time = pd.date_range("2020-01-01", periods=N_TIME, freq="6h")
    lon = np.arange(N_LON) * 360.0 / N_LON
    lat = -90.0 + (np.arange(N_LAT) + 0.5) * 180.0 / N_LAT
    level = np.array([500, 850])
    rng = np.random.default_rng(0)

    def levels_var():
        return (("time", "level", "longitude", "latitude"),
                rng.normal(size=(N_TIME, 2, N_LON, N_LAT)))

    def surface_var():
        return (("time", "longitude", "latitude"),
                rng.normal(size=(N_TIME, N_LON, N_LAT)))

    return xr.Dataset(
        {
            "temperature": levels_var(),
            "geopotential": levels_var(),
            "10m_u_component_of_wind": surface_var(),
            "10m_v_component_of_wind": surface_var(),
            "2m_temperature": surface_var(),
        },
        coords={"time": time, "level": level, "longitude": lon, "latitude": lat},
    )


class TestSelectVariables:
    """Tests for picking model channels out of a WeatherBench2 dataset."""

    def test_names_order_and_levels(self):
        ds = wb2_like_dataset()
        out = select_variables(ds, VARIABLES)

        assert list(out.data_vars) == ["t850", "z500", "u10", "v10"]
        np.testing.assert_array_equal(
            out["t850"].values, ds["temperature"].sel(level=850).values
        )
        np.testing.assert_array_equal(
            out["u10"].values, ds["10m_u_component_of_wind"].values
        )
        assert "level" not in out.dims

    def test_missing_variable_raises(self):
        with pytest.raises(KeyError):
            select_variables(wb2_like_dataset(), {"q700": {"name": "specific_humidity"}})

    def test_missing_level_raises(self):
        with pytest.raises(KeyError):
            select_variables(wb2_like_dataset(), {"t1": {"name": "temperature", "level": 1}})

    def test_level_required_for_pressure_variables(self):
        with pytest.raises(ValueError):
            select_variables(wb2_like_dataset(), {"t": {"name": "temperature"}})


class TestCreateInputTargetPairs:
    """Tests for (X(t), Y(t + lead)) pair creation."""

    def test_output_is_channels_lat_lon(self):
        ds = select_variables(wb2_like_dataset(), VARIABLES)
        inputs, targets = create_input_target_pairs(ds, lead_time_steps=1)

        assert inputs.shape == (N_TIME - 1, 4, N_LAT, N_LON)
        # Channel 0 at (time 0, lat 1, lon 2) must be t850 at that lat/lon
        expected = ds["t850"].isel(time=0).sel(
            latitude=ds.latitude[1], longitude=ds.longitude[2]
        )
        assert inputs[0, 0, 1, 2] == pytest.approx(float(expected))

    def test_independent_of_storage_dim_order(self):
        ds = select_variables(wb2_like_dataset(), VARIABLES)
        reordered = ds.transpose("time", "latitude", "longitude")

        a, _ = create_input_target_pairs(ds)
        b, _ = create_input_target_pairs(reordered)
        np.testing.assert_array_equal(a, b)

    def test_lead_time_offset(self):
        ds = select_variables(wb2_like_dataset(), VARIABLES)
        inputs, targets = create_input_target_pairs(ds, lead_time_steps=2)

        assert inputs.shape[0] == N_TIME - 2
        np.testing.assert_array_equal(targets[0], inputs[2])

    def test_target_times_match_targets(self):
        ds = select_variables(wb2_like_dataset(), VARIABLES)
        _, targets = create_input_target_pairs(ds, lead_time_steps=2)
        times = target_times(ds, lead_time_steps=2)

        assert len(times) == len(targets)
        # Target 0 is the t850 field at times[0]
        at_time = ds["t850"].sel(time=times[0]).transpose("latitude", "longitude")
        np.testing.assert_array_equal(targets[0, 0], at_time.values)


class TestGridAndLeadTime:
    """Tests for grid-convention and lead-time checks."""

    def test_wb2_grid_passes(self):
        check_grid(wb2_like_dataset())

    def test_descending_latitude_raises(self):
        ds = wb2_like_dataset()
        with pytest.raises(ValueError):
            check_grid(ds.isel(latitude=slice(None, None, -1)))

    def test_minus180_longitudes_raise(self):
        ds = wb2_like_dataset()
        ds = ds.assign_coords(longitude=ds.longitude.values - 180.0)
        with pytest.raises(ValueError):
            check_grid(ds)

    @pytest.mark.parametrize("hours, steps", [(6, 1), (12, 2), (24, 4)])
    def test_lead_time_to_steps(self, hours, steps):
        assert lead_time_to_steps(wb2_like_dataset(), hours) == steps

    @pytest.mark.parametrize("hours", [0, 3, 9])
    def test_invalid_lead_time_raises(self, hours):
        with pytest.raises(ValueError):
            lead_time_to_steps(wb2_like_dataset(), hours)
