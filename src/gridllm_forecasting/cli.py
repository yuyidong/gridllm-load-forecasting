from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gridllm_forecasting.data import load_load_series
from gridllm_forecasting.models import PromptForecastSpec, build_forecast_prompt
from gridllm_forecasting.pipeline import load_config, run_from_config


def _run(args: argparse.Namespace) -> None:
    metrics = run_from_config(args.config)
    print(json.dumps(metrics, indent=2))


def _prompt(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    base_dir = Path(args.config).resolve().parent.parent
    data_cfg = config["data"]
    data_path = Path(data_cfg["path"])
    if not data_path.is_absolute():
        data_path = base_dir / data_path

    timestamp_col = data_cfg.get("timestamp_col", "timestamp")
    value_col = data_cfg.get("value_col", "load_kw")
    frame = load_load_series(data_path, timestamp_col, value_col)
    history = frame[value_col].tail(args.last_hours).to_numpy(dtype=float)
    spec = PromptForecastSpec(
        horizon=int(config.get("forecast", {}).get("horizon", 24)),
        unit=args.unit,
        site_description=args.site,
    )
    print(build_forecast_prompt(np.asarray(history), spec))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GridLLM short-term load forecasting toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a configured rolling forecast experiment")
    run_parser.add_argument("--config", required=True, help="Path to YAML config")
    run_parser.set_defaults(func=_run)

    prompt_parser = subparsers.add_parser("prompt", help="Build a strict JSON forecast prompt for an LLM")
    prompt_parser.add_argument("--config", required=True, help="Path to YAML config")
    prompt_parser.add_argument("--last-hours", type=int, default=72)
    prompt_parser.add_argument("--unit", default="kW")
    prompt_parser.add_argument("--site", default="building electricity load")
    prompt_parser.set_defaults(func=_prompt)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

