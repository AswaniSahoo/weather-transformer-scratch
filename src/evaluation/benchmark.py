"""
Benchmark against published forecasts hosted by WeatherBench2.

WeatherBench2 (Rasp et al., 2023) hosts forecasts from leading weather models,
regridded to the same 5.625° grid as this project. This module scores them with
the same truth (ERA5), metrics, climatology and regions as evaluate.py, next to
persistence, anomaly persistence and, if a checkpoint is given, this project's
model rolled forward 6 hours at a time.

Those models run at 0.25° or finer and lose detail when averaged onto the
5.625° grid, so the scores show their skill at this coarse resolution, not
their full skill.

Usage:
    python -m src.evaluation.benchmark --year 2020
    python -m src.evaluation.benchmark --year 2022 --models fgn
    python -m src.evaluation.benchmark --year 2020 --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import xarray as xr
import yaml

from src.data.download import select_variables
from src.data.preprocessing import check_grid
from src.evaluation.evaluate import (
    load_model,
    load_normalization_stats,
    load_target_times,
    regional_scores,
    score,
    seasonal_climatology,
)
from src.evaluation.metrics import latitude_weights
from src.evaluation.regions import SEASONS

WB2 = "gs://weatherbench2/datasets"

# ERA5 on the same grid. The project's training store ends on 31 December 2021;
# this one runs to 10 January 2023, so it also covers FGN's 2022 (same values for 2020).
ERA5 = f"{WB2}/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"

# Forecasts on the 64x32 equiangular grid, initialized at 00 and 12 UTC (FGN: every 6 h)
STORES = {
    "graphcast": f"{WB2}/graphcast/2020/"
                 "date_range_2019-11-16_2021-02-01_12_hours-64x32_equiangular_conservative.zarr",
    "hres": f"{WB2}/hres/2016-2022-0012-64x32_equiangular_conservative.zarr",
    "pangu": f"{WB2}/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr",
    "fuxi": f"{WB2}/fuxi/2020-64x32_equiangular_conservative.zarr",
    "neuralgcm": f"{WB2}/neuralgcm_deterministic/2020-64x32_equiangular_conservative.zarr",
    # Ensemble mean of FGN, the model behind Google DeepMind's WeatherNext 2
    "fgn": f"{WB2}/fgn/2022-64x32_equiangular_conservative_mean.zarr",
}

LEAD_HOURS = (6, 12, 24, 48, 72)

# Drawn in the --figure plot when present, next to this model and the baselines:
# the strongest ERA5-trained model, the operational physics model, and FGN
FIGURE_MODELS = ("graphcast", "hres", "fgn")


def open_store(path: str) -> xr.Dataset:
    """Open a public WeatherBench2 Zarr store (anonymous access)."""
    import gcsfs

    return xr.open_zarr(gcsfs.GCSFileSystem(token="anon").get_mapper(path), decode_timedelta=True)


def _hours(times: np.ndarray) -> np.ndarray:
    return ((times - times.astype("datetime64[D]")) / np.timedelta64(1, "h")).astype(int)


def common_inits(stores: Sequence[xr.Dataset], year: int, max_lead_hours: int) -> np.ndarray:
    """
    00 and 12 UTC initialization times in `year` that every store has and
    whose longest forecast still ends within the year.
    """
    last_valid = np.datetime64(f"{year}-12-31T18")
    inits = None
    for ds in stores:
        t = ds.time.values
        t = t[(t >= np.datetime64(f"{year}-01-01"))
              & (t + np.timedelta64(max_lead_hours, "h") <= last_valid)]
        t = t[np.isin(_hours(t), (0, 12))]
        inits = t if inits is None else np.intersect1d(inits, t)
    return inits


def available_leads(ds: xr.Dataset, lead_hours: Sequence[int]) -> List[int]:
    """The lead times (hours) from `lead_hours` that the store provides."""
    have = set((ds.prediction_timedelta.values / np.timedelta64(1, "h")).astype(int).tolist())
    return [h for h in lead_hours if h in have]


def forecast_fields(
    ds: xr.Dataset,
    variables: Dict[str, dict],
    inits: np.ndarray,
    lead_hours: Sequence[int],
) -> Tuple[np.ndarray, List[str]]:
    """
    Forecasts for the given initialization and lead times, in physical units.

    Args:
        ds: WeatherBench2 forecast store (time, prediction_timedelta, ...).
        variables: Channel mapping as in configs/default.yaml.
        inits: Initialization times to select.
        lead_hours: Lead times to select; all must be in the store.

    Returns:
        (I, L, C, H, W) float32 array with latitude ascending, as the model
        sees it, and the names of the C channels the store provides.
    """
    names = [name for name, spec in variables.items() if spec["name"] in ds]
    subset = ds.sel(time=inits, prediction_timedelta=[np.timedelta64(h, "h") for h in lead_hours])
    selected = select_variables(subset, {name: variables[name] for name in names})
    fields = np.stack(
        [selected[name].transpose("time", "prediction_timedelta", "latitude", "longitude").values
         for name in names],
        axis=2,
    )
    return fields.astype(np.float32), names


def era5_fields(
    variables: Dict[str, dict], start: np.datetime64, end: np.datetime64
) -> Tuple[np.ndarray, np.ndarray]:
    """ERA5 states from `start` to `end` inclusive: (T, C, H, W) in physical units, and times."""
    ds = open_store(ERA5).sel(time=slice(start, end))
    check_grid(ds)
    selected = select_variables(ds, variables)
    fields = np.stack(
        [selected[name].transpose("time", "latitude", "longitude").values for name in variables],
        axis=1,
    )
    return fields.astype(np.float32), ds.time.values


def normalize(fields: np.ndarray, names: Sequence[str], stats: dict, channel_axis: int):
    """(x - mean) / std per channel, with the training-years statistics."""
    shape = [1] * fields.ndim
    shape[channel_axis] = len(names)
    mean = np.array([stats[n]["mean"] for n in names], dtype=np.float32).reshape(shape)
    std = np.array([stats[n]["std"] for n in names], dtype=np.float32).reshape(shape)
    return torch.from_numpy((fields - mean) / std)


@torch.no_grad()
def rollout(model, x0: torch.Tensor, steps: int, device: str, batch_size: int = 64):
    """
    Feed the model its own forecast `steps` times, starting from x0 (N, C, H, W).

    Returns:
        (steps, N, C, H, W): the forecast after 1, 2, ..., steps model steps.
    """
    batches = []
    for x in x0.split(batch_size):
        x = x.to(device)
        states = []
        for _ in range(steps):
            x = model(x)
            states.append(x.cpu())
        batches.append(torch.stack(states))
    return torch.cat(batches, dim=1)


def benchmark(
    truth: torch.Tensor,
    truth_times: np.ndarray,
    forecasts: Dict[str, Tuple[torch.Tensor, List[int], List[str]]],
    inits: np.ndarray,
    climatology_at: Callable[[np.ndarray], torch.Tensor],
    variable_names: List[str],
    stats: dict,
) -> Dict[str, dict]:
    """
    Score every forecast, persistence and anomaly persistence at every lead time.

    Args:
        truth: (T, C, H, W) normalized ERA5 states.
        truth_times: (T,) times of `truth`, 6-hourly.
        forecasts: name -> ((I, L, C', H, W) normalized forecasts, the L lead
            times in hours, the names of the C' channels).
        inits: (I,) initialization times shared by all forecasts.
        climatology_at: times -> (N, C, H, W) normalized climatology.
        variable_names: Names of the C channels of `truth`.
        stats: Normalization stats, for RMSE in physical units.

    Returns:
        {lead hours: {"global" or "<region>/<season>": {forecast name: metrics}}}.
    """
    weights = latitude_weights(truth.shape[-2])
    init_index = _index_of(truth_times, inits)
    results = {}

    for lead in sorted({h for _, leads, _ in forecasts.values() for h in leads}):
        valid = inits + np.timedelta64(lead, "h")
        inputs, targets = truth[init_index], truth[_index_of(truth_times, valid)]
        climatology, input_climatology = climatology_at(valid), climatology_at(inits)

        baselines = _by_area(
            score(inputs, inputs, targets, variable_names, weights, climatology,
                  input_climatology, stats),
            regional_scores(inputs, inputs, targets, valid, climatology, variable_names,
                            input_climatology, stats),
        )
        by_area = {
            area: {name: scores[name] for name in ("persistence", "anomaly_persistence")}
            for area, scores in baselines.items()
        }

        for name, (fields, leads, channels) in forecasts.items():
            if lead not in leads:
                continue
            idx = [variable_names.index(c) for c in channels]
            predictions = fields[:, leads.index(lead)]
            scored = _by_area(
                score(inputs[:, idx], predictions, targets[:, idx], channels, weights,
                      climatology[:, idx], stats=stats),
                regional_scores(inputs[:, idx], predictions, targets[:, idx], valid,
                                climatology[:, idx], channels, stats=stats),
            )
            for area, scores in scored.items():
                by_area[area][name] = scores["model"]

        results[str(lead)] = by_area

    return results


def _index_of(all_times: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Positions of `times` in the sorted `all_times`; every time must be present."""
    idx = np.searchsorted(all_times, times)
    if idx.max() >= len(all_times) or not np.array_equal(all_times[idx], times):
        raise ValueError("ERA5 truth does not cover all initialization and valid times")
    return idx


def _by_area(global_scores: dict, regions: dict) -> Dict[str, dict]:
    """{"global": ..., "<region>/<season>": ...} from score() and regional_scores()."""
    areas = {"global": global_scores}
    for region, by_season in regions.items():
        for season in SEASONS:
            if by_season[season]["samples"]:
                areas[f"{region}/{season}"] = by_season[season]
    return areas


def load_forecasts(
    name: str,
    ds: xr.Dataset,
    variables: Dict[str, dict],
    inits: np.ndarray,
    stats: dict,
    cache_dir: str,
) -> Tuple[torch.Tensor, List[int], List[str]]:
    """Normalized forecasts from a store; downloaded once, then read from cache_dir."""
    leads = available_leads(ds, LEAD_HOURS)
    cache = Path(cache_dir) / f"{name}_{str(inits[0])[:4]}.npz"
    if cache.exists():
        saved = np.load(cache)
        if np.array_equal(saved["inits"], inits) and saved["leads"].tolist() == leads:
            names = saved["names"].tolist()
            return normalize(saved["fields"], names, stats, channel_axis=2), leads, names

    print(f"Downloading {name}: {len(inits)} forecasts x {len(leads)} lead times")
    fields, names = forecast_fields(ds, variables, inits, leads)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, fields=fields, names=names, inits=inits, leads=leads)
    return normalize(fields, names, stats, channel_axis=2), leads, names


def markdown_summary(results: Dict[str, dict], variable_names: List[str]) -> str:
    """RMSE (physical units) and ACC tables: one row per forecast, one column per lead time."""
    leads = list(results)
    lines = []
    for area in results[leads[0]]:
        names = list(dict.fromkeys(n for h in leads for n in results[h].get(area, {})))
        for var in variable_names:
            for label, key, fmt in ((f"{var} RMSE", f"rmse_{var}_physical", ".3g"),
                                    (f"{var} ACC", f"acc_{var}", ".3f")):
                lines += [f"### {area}: {label}", "",
                          "| Forecast | " + " | ".join(f"{h} h" for h in leads) + " |",
                          "|---|" + "---:|" * len(leads)]
                for name in names:
                    cells = [results[h].get(area, {}).get(name, {}).get(key) for h in leads]
                    lines.append(f"| {name} | " + " | ".join(
                        "" if c is None else format(c, fmt) for c in cells) + " |")
                lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark against WeatherBench2 forecasts")
    parser.add_argument("--year", type=int, default=2020, help="Verification year")
    parser.add_argument(
        "--models", nargs="+", choices=sorted(STORES),
        default=["graphcast", "hres", "pangu", "fuxi", "neuralgcm"],
        help="WeatherBench2 forecasts to score (FGN covers 2022 only)",
    )
    parser.add_argument("--checkpoint", default=None, help="Also roll out this project's model")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument(
        "--data-dir", default="data/processed",
        help="Processed data: training targets and times, normalization stats",
    )
    parser.add_argument("--cache-dir", default="data/benchmark", help="Downloaded forecasts")
    parser.add_argument("--output", default=None, help="Default: results/benchmark_<year>.json")
    parser.add_argument(
        "--figure", default=None,
        help="Also plot RMSE against lead time (global and India JJAS) to this PNG",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    variables = config["data"]["variables"]
    names = list(variables)
    stats = load_normalization_stats(args.data_dir)
    train_targets = np.load(Path(args.data_dir) / "train_targets.npy", mmap_mode="r")
    train_times = load_target_times(args.data_dir, "train")

    stores = {name: open_store(STORES[name]) for name in args.models}
    inits = common_inits(list(stores.values()), args.year, max(LEAD_HOURS))
    print(f"{len(inits)} shared initializations, {inits[0]} to {inits[-1]}")

    end = inits[-1] + np.timedelta64(max(LEAD_HOURS), "h")
    truth_fields, truth_times = era5_fields(variables, inits[0], end)
    truth = normalize(truth_fields, names, stats, channel_axis=1)

    forecasts = {
        name: load_forecasts(name, ds, variables, inits, stats, args.cache_dir)
        for name, ds in stores.items()
    }

    if args.checkpoint:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(args.checkpoint, config, device)
        step = config["data"].get("lead_time_hours", 6)
        states = rollout(model, truth[_index_of(truth_times, inits)], max(LEAD_HOURS) // step, device)
        at_leads = states[[h // step - 1 for h in LEAD_HOURS]].transpose(0, 1)
        forecasts["this_model"] = (at_leads, list(LEAD_HOURS), names)

    def climatology_at(times):
        return seasonal_climatology(train_targets, train_times, times)

    results = benchmark(truth, truth_times, forecasts, inits, climatology_at, names, stats)

    output = Path(args.output or f"results/benchmark_{args.year}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump({
            "year": args.year,
            "initializations": len(inits),
            "first_init": str(inits[0]),
            "last_init": str(inits[-1]),
            "truth": "ERA5, WeatherBench2 64x32 equiangular grid",
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    summary = output.with_suffix(".md")
    summary.write_text(markdown_summary(results, names), encoding="utf-8")
    print(f"Saved {output} and {summary}")

    if args.figure:
        from src.visualization.plot_benchmark import plot_error_by_lead

        drawn = [n for n in ("this_model", *FIGURE_MODELS) if n in forecasts]
        plot_error_by_lead(
            results, ["global", "india/monsoon_jjas"], ["t850", "z500"],
            drawn + ["persistence", "anomaly_persistence"], save_path=args.figure,
            title=f"6-hour to 3-day forecast error, {args.year} (5.625° grid, truth: ERA5)",
        )
        print(f"Saved {args.figure}")


if __name__ == "__main__":
    main()
