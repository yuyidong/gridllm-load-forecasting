from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Forecaster(Protocol):
    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """Predict the next horizon values from a 1-D context sequence."""


@dataclass
class SeasonalNaiveForecaster:
    season_length: int = 24

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        history = np.asarray(context, dtype=float)
        if history.size == 0:
            raise ValueError("Context cannot be empty.")

        if history.size >= self.season_length:
            pattern = history[-self.season_length :]
        else:
            pattern = history
        return np.resize(pattern, horizon).astype(float)


@dataclass
class MovingAverageForecaster:
    window: int = 24

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        history = np.asarray(context, dtype=float)
        if history.size == 0:
            raise ValueError("Context cannot be empty.")
        value = float(np.mean(history[-self.window :]))
        return np.repeat(value, horizon)


@dataclass
class PromptForecastSpec:
    horizon: int
    unit: str = "kW"
    granularity: str = "hourly"
    site_description: str = "building electricity load"


def build_forecast_prompt(context: np.ndarray, spec: PromptForecastSpec) -> str:
    recent_values = [round(float(value), 4) for value in np.asarray(context, dtype=float)]
    return (
        "You are an expert short-term electricity load forecasting model.\n"
        f"Task: forecast the next {spec.horizon} {spec.granularity} load values.\n"
        f"Site: {spec.site_description}.\n"
        f"Unit: {spec.unit}.\n"
        "Use daily seasonality, recent trend, and peak/off-peak structure.\n"
        "Return strict JSON only, with this schema: {\"forecast\": [number, ...]}.\n"
        f"Historical load sequence: {json.dumps(recent_values)}"
    )


def parse_llm_json_forecast(text: str, horizon: int) -> np.ndarray:
    payload = json.loads(text)
    values = payload.get("forecast")
    if not isinstance(values, list):
        raise ValueError("LLM response must contain a 'forecast' list.")
    forecast = np.asarray(values, dtype=float)
    if forecast.size != horizon:
        raise ValueError(f"Expected {horizon} forecast values, got {forecast.size}.")
    return forecast


def make_forecaster(config: dict) -> Forecaster:
    name = str(config.get("name", "seasonal_naive")).lower()
    if name == "seasonal_naive":
        return SeasonalNaiveForecaster(season_length=int(config.get("season_length", 24)))
    if name == "moving_average":
        return MovingAverageForecaster(window=int(config.get("window", 24)))
    raise ValueError(f"Unknown model: {name}")
