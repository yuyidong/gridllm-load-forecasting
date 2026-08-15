from pathlib import Path

from gridllm_forecasting.pipeline import run_from_config


def test_example_pipeline_runs() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "example.yaml"
    metrics = run_from_config(config_path)

    assert set(metrics) == {"mae", "rmse", "mape", "nrmse"}
    assert metrics["rmse"] >= 0

