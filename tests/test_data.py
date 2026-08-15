import pandas as pd

from gridllm_forecasting.data import make_rolling_windows


def test_make_rolling_windows_supports_non_default_index() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
            "load_kw": [0.0, 1.0, 2.0, 3.0],
        },
        index=[10, 20, 30, 40],
    )

    [window] = make_rolling_windows(
        frame,
        timestamp_col="timestamp",
        value_col="load_kw",
        context_length=2,
        horizon=2,
        stride=1,
    )

    assert window.target.tolist() == [2.0, 3.0]
    expected = frame["timestamp"].iloc[2:4].reset_index(drop=True)
    pd.testing.assert_series_equal(window.target_timestamps, expected)
