from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastWindow:
    context: np.ndarray
    target: np.ndarray
    target_timestamps: pd.Series


def load_load_series(path: Union[str, Path], timestamp_col: str, value_col: str) -> pd.DataFrame:
    """Load and normalize a univariate electricity-load CSV."""
    frame = pd.read_csv(path)
    missing = {timestamp_col, value_col}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    result = frame[[timestamp_col, value_col]].copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col], errors="coerce")
    result[value_col] = pd.to_numeric(result[value_col], errors="coerce")
    result = result.dropna(subset=[timestamp_col, value_col]).sort_values(timestamp_col)
    result = result.drop_duplicates(subset=[timestamp_col], keep="last").reset_index(drop=True)

    if result.empty:
        raise ValueError("No valid load observations were found.")
    return result


def make_rolling_windows(
    frame: pd.DataFrame,
    timestamp_col: str,
    value_col: str,
    context_length: int,
    horizon: int,
    stride: int,
) -> list[ForecastWindow]:
    if context_length <= 0 or horizon <= 0 or stride <= 0:
        raise ValueError("context_length, horizon, and stride must all be positive.")

    values = frame[value_col].to_numpy(dtype=float)
    windows: list[ForecastWindow] = []
    last_start = len(values) - context_length - horizon
    for start in range(0, last_start + 1, stride):
        context_end = start + context_length
        target_end = context_end + horizon
        windows.append(
            ForecastWindow(
                context=values[start:context_end],
                target=values[context_end:target_end],
                target_timestamps=frame[timestamp_col]
                .iloc[context_end:target_end]
                .reset_index(drop=True),
            )
        )

    if not windows:
        raise ValueError(
            "Not enough observations to create one forecast window. "
            f"Need at least {context_length + horizon}, got {len(values)}."
        )
    return windows
