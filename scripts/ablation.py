"""
Step 6 ablation: does predicting the 6-hour change help, and do the
smoothness and conservation loss terms help?

Trains every variant with several seeds, scores each best checkpoint on the
test set with src/evaluation/evaluate.py, and summarizes the mean and spread
over seeds. Each run has its own directory with its config, so an
interrupted ablation continues where it stopped.

Variants (changes to configs/default.yaml):
    baseline           predict the full state; MSE + smoothness + conservation
    tendency           predict the 6-hour change (model.residual); same loss
    tendency_mse       predict the 6-hour change; MSE only
    tendency_mse_area  predict the 6-hour change; MSE weighted by cell area,
                       the same weighting the evaluation uses

Usage:
    python scripts/ablation.py --seeds 42 43 44 --device cuda
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

VARIANTS = {
    "baseline": {},
    "tendency": {"model": {"residual": True}},
    "tendency_mse": {
        "model": {"residual": True},
        "loss": {"smoothness_weight": 0.0, "conservation_weight": 0.0},
    },
    "tendency_mse_area": {
        "model": {"residual": True},
        "loss": {"smoothness_weight": 0.0, "conservation_weight": 0.0, "latitude_weighted": True},
    },
}
VARIABLES = ["t850", "z500", "u10", "v10"]
UNITS = {"t850": "K", "z500": "m²/s²", "u10": "m/s", "v10": "m/s"}
# The forecasts to summarize, for the whole globe and for India in the monsoon
AREAS = {
    "global": lambda metrics: metrics,
    "india_jjas": lambda metrics: metrics["regions"]["india"]["monsoon_jjas"],
}


def variant_config(base: dict, changes: dict, seed: int, run_dir: Path) -> dict:
    """The base config with one variant's changes, a seed and its own checkpoint directory."""
    config = copy.deepcopy(base)
    for section, values in changes.items():
        config[section].update(values)
    config["training"]["seed"] = seed
    config["training"]["checkpoint_dir"] = str(run_dir)
    return config


def summarize(runs: Dict[str, List[dict]]) -> dict:
    """
    Mean and standard deviation over seeds of every variable's RMSE and ACC.

    Args:
        runs: variant -> metrics.json contents of its runs, one per seed.

    Returns:
        {area: {variant or baseline: {"seeds", metric: {"mean", "std"}}}}.
        The baselines do not depend on the seed, so they come from one run.
    """
    keys = [f"rmse_{v}_physical" for v in VARIABLES] + [f"acc_{v}" for v in VARIABLES]
    any_run = next(iter(runs.values()))[0]
    summary = {}
    for area, pick in AREAS.items():
        summary[area] = {}
        for variant, metrics in runs.items():
            values = {k: [pick(m)["model"][k] for m in metrics] for k in keys}
            summary[area][variant] = {"seeds": len(metrics), **{
                k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                for k, v in values.items()
            }}
        for baseline in ("persistence", "anomaly_persistence"):
            scores = pick(any_run)[baseline]
            summary[area][baseline] = {"seeds": 0, **{k: {"mean": scores[k], "std": 0.0} for k in keys}}
    return summary


def markdown(summary: dict) -> str:
    """One table per area: RMSE and ACC per variable, mean ± standard deviation over seeds."""
    lines = []
    for area, rows in summary.items():
        lines += [
            f"### {area}", "",
            "| Forecast | Seeds | " + " | ".join(f"{v} RMSE ({UNITS[v]})" for v in VARIABLES)
            + " | " + " | ".join(f"{v} ACC" for v in VARIABLES) + " |",
            "|---|---:|" + "---:|" * (2 * len(VARIABLES)),
        ]
        for name, row in rows.items():
            cells = []
            for v in VARIABLES:
                m = row[f"rmse_{v}_physical"]
                digits = 0 if v == "z500" else 3
                cells.append(f"{m['mean']:.{digits}f} ± {m['std']:.{digits}f}" if row["seeds"] > 1
                             else f"{m['mean']:.{digits}f}")
            for v in VARIABLES:
                m = row[f"acc_{v}"]
                cells.append(f"{m['mean']:.3f} ± {m['std']:.3f}" if row["seeds"] > 1
                             else f"{m['mean']:.3f}")
            lines.append(f"| {name} | {row['seeds'] or '–'} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Tendency-prediction ablation (roadmap step 6)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--config", default="configs/default.yaml", help="Base config")
    parser.add_argument("--data-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--runs-dir", default="runs/ablation", help="One subdirectory per run")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--output", default="results/ablation.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base = yaml.safe_load(f)

    runs = {}
    for variant in args.variants:
        runs[variant] = []
        for seed in args.seeds:
            run_dir = Path(args.runs_dir) / variant / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / "config.yaml"
            config = variant_config(base, VARIANTS[variant], seed, run_dir)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))

            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                train = [sys.executable, "scripts/train.py", "--config", str(config_path),
                         "--data-dir", args.data_dir, "--resume"]
                if args.device:
                    train += ["--device", args.device]
                subprocess.run(train, check=True)
                subprocess.run([
                    sys.executable, "-m", "src.evaluation.evaluate",
                    "--checkpoint", str(run_dir / "best_model.pt"), "--config", str(config_path),
                    "--data-dir", args.data_dir, "--output", str(metrics_path),
                ], check=True)
            runs[variant].append(json.loads(metrics_path.read_text(encoding="utf-8")))

    summary = summarize(runs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"seeds": args.seeds, "variants": VARIANTS, "summary": summary}, f, indent=2,
                  ensure_ascii=False)
    output.with_suffix(".md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))


if __name__ == "__main__":
    main()
