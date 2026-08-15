import numpy as np

from gridllm_forecasting.metrics import mae, nrmse, rmse


def test_basic_metrics() -> None:
    true = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 4.0])

    assert mae(true, pred) == 1 / 3
    assert round(rmse(true, pred), 4) == 0.5774
    assert round(nrmse(true, pred), 4) == 28.8675

