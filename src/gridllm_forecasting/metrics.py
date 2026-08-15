from __future__ import annotations

import numpy as np


def _arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={true.shape}, y_pred={pred.shape}")
    return true, pred


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(true - pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    true, pred = _arrays(y_true, y_pred)
    denom = np.maximum(np.abs(true), epsilon)
    return float(np.mean(np.abs((true - pred) / denom)) * 100)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    true, pred = _arrays(y_true, y_pred)
    scale = max(float(np.mean(np.abs(true))), epsilon)
    return rmse(true, pred) / scale * 100


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
    }

