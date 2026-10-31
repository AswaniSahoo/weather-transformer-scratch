"""
Unit tests for the WeatherBench2 benchmark, on synthetic stores (no network).

Run with: pytest tests/test_benchmark.py -v
"""

import numpy as np
import pandas as pd
import pytest
import torch

xr = pytest.importorskip("xarray")

from src.evaluation import benchmark  # noqa: E402

VARIABLES = {
    "t850": {"name": "temperature", "level": 850},
    "z500": {"name": "geopotential", "level": 500},
    "u10": {"name": "10m_u_component_of_wind"},
    "v10": {"name": "10m_v_component_of_wind"},
}
N_LON, N_LAT = 8, 4
LEADS = np.array([6, 12, 24], dtype="timedelta64[h]").astype("timedelta64[ns]")


def wb2_like_forecasts(times: np.ndarray, with_wind: bool = True) -> xr.Dataset:
    """Forecast store with WeatherBench2's dims; each value encodes its position."""
    t, lead, level, lon, lat = np.meshgrid(
        np.arange(len(times)), np.arange(len(LEADS)), np.arange(2), np.arange(N_LON),
        np.arange(N_LAT), indexing="ij",
    )
    code = (10000 * t + 1000 * lead + 100 * level + 10 * lon + lat).astype(float)
    dims = ("time", "prediction_timedelta", "level", "longitude", "latitude")
    data = {"temperature": (dims, code), "geopotential": (dims, -code)}
    if with_wind:
        surface = dims[:2] + dims[3:]
        data["10m_u_component_of_wind"] = (surface, code[:, :, 0] + 0.5)
        data["10m_v_component_of_wind"] = (surface, code[:, :, 0] + 0.25)
    return xr.Dataset(data, coords={
        "time": times,
        "prediction_timedelta": LEADS,
        "level": [500, 850],
        "longitude": np.arange(N_LON) * 45.0,
        "latitude": -67.5 + 45.0 * np.arange(N_LAT),
    })


def twelve_hourly(start: str, periods: int) -> np.ndarray:
    return pd.date_range(start, periods=periods, freq="12h").values


class TestForecastFields:
    def test_values_land_at_init_lead_channel_lat_lon(self):
        times = twelve_hourly("2020-01-01", 4)
        ds = wb2_like_forecasts(times)

        fields, names = benchmark.forecast_fields(ds, VARIABLES, times[[1, 2]], [12, 24])

        assert names == ["t850", "z500", "u10", "v10"]
        assert fields.shape == (2, 2, 4, N_LAT, N_LON)
        # init 1, 24 h lead (index 2), latitude 3, longitude 5
        assert fields[0, 1, 0, 3, 5] == 10000 * 1 + 1000 * 2 + 100 * 1 + 10 * 5 + 3  # t850
        assert fields[0, 1, 1, 3, 5] == -(10000 * 1 + 1000 * 2 + 100 * 0 + 10 * 5 + 3)  # z500

    def test_store_on_a_different_grid_is_refused(self):
        times = twelve_hourly("2020-01-01", 2)
        flipped = wb2_like_forecasts(times).isel(latitude=slice(None, None, -1))
        with pytest.raises(ValueError):
            benchmark.forecast_fields(flipped, VARIABLES, times, [6])

    def test_missing_variables_are_left_out(self):
        times = twelve_hourly("2020-01-01", 2)
        _, names = benchmark.forecast_fields(
            wb2_like_forecasts(times, with_wind=False), VARIABLES, times, [6]
        )
        assert names == ["t850", "z500"]


def test_available_leads():
    ds = wb2_like_forecasts(twelve_hourly("2020-01-01", 1))
    assert benchmark.available_leads(ds, (6, 12, 24, 48)) == [6, 12, 24]


def test_common_inits_are_shared_00_and_12_utc_within_the_year():
    every_6h = wb2_like_forecasts(pd.date_range("2020-12-25", "2020-12-31T18", freq="6h").values)
    every_12h = wb2_like_forecasts(twelve_hourly("2020-12-20", 20))  # to 29 Dec 12 UTC

    inits = benchmark.common_inits([every_6h, every_12h], 2020, max_lead_hours=24)

    np.testing.assert_array_equal(inits, twelve_hourly("2020-12-25", 10))


class AddOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


def test_rollout_feeds_the_model_its_own_forecast():
    states = benchmark.rollout(AddOne(), torch.zeros(5, 1, 2, 2), steps=3, device="cpu",
                               batch_size=2)

    assert states.shape == (3, 5, 1, 2, 2)
    assert states[:, :, 0, 0, 0].tolist() == [[1.0] * 5, [2.0] * 5, [3.0] * 5]


def test_benchmark_scores_forecasts_and_baselines():
    times = pd.date_range("2020-06-01", periods=40, freq="6h").values
    truth = torch.randn(40, 4, 32, 64)
    inits = times[[0, 2, 4, 6]]
    idx = np.searchsorted(times, inits)
    perfect = torch.stack([truth[idx + 1], truth[idx + 2]], dim=1)  # 6 h and 12 h leads
    forecasts = {
        "perfect": (perfect, [6, 12], list(VARIABLES)),
        "t850_only": (perfect[:, 1:, :1], [12], ["t850"]),
    }
    stats = {name: {"mean": 0.0, "std": 2.0} for name in VARIABLES}

    results = benchmark.benchmark(
        truth, times, forecasts, inits, lambda t: torch.zeros(len(t), 4, 32, 64),
        list(VARIABLES), stats,
    )

    assert set(results) == {"6", "12"}
    at_6h = results["6"]["global"]
    assert at_6h["perfect"]["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert at_6h["persistence"]["rmse_t850_physical"] == pytest.approx(
        2.0 * at_6h["persistence"]["rmse_t850"]
    )
    assert "anomaly_persistence" in at_6h
    assert "t850_only" not in at_6h
    assert set(results["12"]["global"]["t850_only"]) >= {"rmse_t850", "acc_t850"}
    assert "rmse_z500" not in results["12"]["global"]["t850_only"]
    assert "india/monsoon_jjas" in results["6"]  # all valid times are in June
    assert "india/rest_of_year" not in results["6"]


def test_benchmark_refuses_times_missing_from_the_truth():
    times = pd.date_range("2020-06-01", periods=4, freq="6h").values
    truth = torch.zeros(4, 4, 32, 64)
    forecasts = {"f": (torch.zeros(1, 1, 4, 32, 64), [24], list(VARIABLES))}

    with pytest.raises(ValueError):
        benchmark.benchmark(truth, times, forecasts, times[:1],
                            lambda t: torch.zeros(len(t), 4, 32, 64), list(VARIABLES), {})


def test_benchmark_figure_is_written(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from src.visualization.plot_benchmark import plot_error_by_lead

    metrics = {"rmse_t850_physical": 1.0}
    results = {h: {"global": {"persistence": metrics, "graphcast": metrics}} for h in ("6", "12")}
    path = tmp_path / "benchmark.png"

    plot_error_by_lead(results, ["global"], ["t850"], ["graphcast", "persistence"], str(path))

    assert path.exists() and path.stat().st_size > 0
