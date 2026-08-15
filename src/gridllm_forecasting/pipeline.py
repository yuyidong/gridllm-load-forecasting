from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml

from gridllm_forecasting.data import load_load_series, make_rolling_windows
from gridllm_forecasting.metrics import summarize_metrics
from gridllm_forecasting.models import make_forecaster


def load_config(path: Union[str, Path]) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_from_config(config_path: Union[str, Path]) -> dict[str, float]:
    config = load_config(config_path)
    base_dir = Path(config_path).resolve().parent.parent

    data_cfg = config["data"]
    forecast_cfg = config["forecast"]
    output_cfg = config.get("output", {})

    data_path = Path(data_cfg["path"])
    if not data_path.is_absolute():
        data_path = base_dir / data_path

    timestamp_col = data_cfg.get("timestamp_col", "timestamp")
    value_col = data_cfg.get("value_col", "load_kw")
    frame = load_load_series(data_path, timestamp_col, value_col)
    windows = make_rolling_windows(
        frame=frame,
        timestamp_col=timestamp_col,
        value_col=value_col,
        context_length=int(forecast_cfg.get("context_length", 168)),
        horizon=int(forecast_cfg.get("horizon", 24)),
        stride=int(forecast_cfg.get("stride", 24)),
    )

    horizon = int(forecast_cfg.get("horizon", 24))
    forecaster = make_forecaster(config.get("model", {}))

    rows: list[dict] = []
    all_true: list[float] = []
    all_pred: list[float] = []

    for window_id, window in enumerate(windows):
        prediction = forecaster.predict(window.context, horizon)
        for step, (timestamp, actual, predicted) in enumerate(
            zip(window.target_timestamps, window.target, prediction), start=1
        ):
            rows.append(
                {
                    "window_id": window_id,
                    "step": step,
                    "timestamp": timestamp,
                    "actual": float(actual),
                    "prediction": float(predicted),
                }
            )
        all_true.extend(window.target.tolist())
        all_pred.extend(prediction.tolist())

    metrics = summarize_metrics(np.asarray(all_true), np.asarray(all_pred))
    output_dir = Path(output_cfg.get("directory", "outputs/demo"))
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(output_dir / "forecast.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics
