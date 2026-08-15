# GridLLM Load Forecasting

GridLLM Load Forecasting 是一个面向电力、园区和建筑能耗场景的短期负荷预测工具包。项目把负荷历史序列、评估指标、滚动预测流程和 LLM Prompt 构造统一到一个轻量 Python 包中，方便快速做预测实验或继续扩展模型。

## 项目能力

- 读取标准 CSV 负荷数据，自动生成滚动预测窗口。
- 支持过去若干小时预测未来若干小时，默认适配日内负荷节律。
- 内置 `seasonal_naive` 和 `moving_average` 两个可离线运行的基线模型。
- 提供 LLM Prompt 构造器，可把历史负荷序列组织成严格 JSON 输出任务。
- 输出 `forecast.csv` 和 `metrics.json`，包含 MAE、RMSE、MAPE、NRMSE。
- 保持包结构清晰，后续可以继续接入深度学习模型或在线大模型接口。

## 快速运行

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
python -m gridllm_forecasting.cli run --config configs/example.yaml
```

运行完成后，结果会写入：

```text
outputs/demo/forecast.csv
outputs/demo/metrics.json
```

## 数据格式

输入数据为 CSV，至少包含时间列和负荷列：

```csv
timestamp,load_kw
2026-01-01 00:00:00,61.2
2026-01-01 01:00:00,58.7
```

字段名可在配置文件中调整：

```yaml
data:
  path: data/sample_load.csv
  timestamp_col: timestamp
  value_col: load_kw
```

## 配置说明

`configs/example.yaml` 提供了一个最小可运行示例：

```yaml
forecast:
  context_length: 48
  horizon: 24
  stride: 24

model:
  name: seasonal_naive
  season_length: 24
```

含义：

- `context_length`：每个预测窗口使用多少个历史点。
- `horizon`：每次预测多少个未来点。
- `stride`：滚动窗口步长。
- `season_length`：季节性重复周期，小时级负荷通常可设为 24。

## 生成 LLM Prompt

```bash
python -m gridllm_forecasting.cli prompt --config configs/example.yaml --last-hours 72
```

该命令会把最近的负荷数据整理成预测任务 Prompt，并要求模型返回严格 JSON：

```json
{"forecast": [70.1, 68.3, 66.9]}
```

## 目录结构

```text
gridllm-load-forecasting/
  configs/
    example.yaml
  data/
    sample_load.csv
  src/
    gridllm_forecasting/
      cli.py
      data.py
      metrics.py
      models.py
      pipeline.py
  tests/
    test_metrics.py
    test_pipeline.py
  PROJECT_GUIDE.md
  PROJECT_LICENSE.txt
  pyproject.toml
```
