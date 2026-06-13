# ts-metric

Time series metric computation library for **prediction**, **imputation**, and **generation** tasks.

Supports both **point** (regression) and **probabilistic** (distribution) evaluation modes.

## 安装

```bash
cd ts_metric
pip install -e .
```

## 输入形状约定

| 场景 | 输入 | Shape |
|------|------|-------|
| Point 指标 | `target`, `forecast` | `(B, C, T)` 或 `(C, T)` |
| Probabilistic 指标 | `target` | `(B, C, T)` 或 `(C, T)` |
| Probabilistic 指标 | `samples` | `(B, S, C, T)` 或 `(S, C, T)` |
| Generation | `real` | `(N, C, T)` 或 `(C, T)` |
| Generation | `generated` | `(M, C, T)` 或 `(C, T)` |

- **B**: batch size
- **C**: number of features (channels)
- **T**: time steps
- **S**: number of probabilistic samples
- **N/M**: number of samples (real vs generated)

## Mask 支持

所有指标支持可选的 `mask` 参数：

- `mask=1` 表示有效时间步（参与计算）
- `mask=0` 表示屏蔽（不参与计算）
- 支持广播：`(T,)`, `(C, T)`, `(B, T)`, `(B, 1, T)`, `(B, C, T)`

## 快速开始

### 函数式 API

```python
import torch
import ts_metric as tm

B, C, T = 4, 3, 24
target = torch.randn(B, C, T)
forecast = target + 0.1 * torch.randn(B, C, T)

# Point metrics
mse_val = tm.prediction.mse(target, forecast)
mae_val = tm.prediction.mae(target, forecast)
r2_val = tm.prediction.r2(target, forecast)

# With mask
mask = torch.ones(B, C, T)
mask[:, :, 12:] = 0  # ignore second half
mse_masked = tm.prediction.mse(target, forecast, mask=mask)

# Probabilistic metrics
samples = target.unsqueeze(1) + 0.2 * torch.randn(B, 50, C, T)  # (B, S, C, T)
crps_val = tm.prediction.crps(target, samples)
picp_val = tm.prediction.picp(target, samples, alpha=0.1)

# Generation metrics
real = torch.randn(100, C, T)
generated = torch.randn(80, C, T)
fid_val = tm.generation.fidelity(real, generated)
mmd_val = tm.generation.mmd(real, generated)
```

### MetricCalculator API

```python
from ts_metric import MetricCalculator

# Select specific metrics
calc = MetricCalculator(
    task="prediction",       # generation | imputation | prediction
    mode="point",            # point | probabilistic
    metrics=["MSE", "RMSE", "MAE", "sMAPE"]
)

results = calc.compute(target, forecast)
# -> {"MSE": tensor(0.01), "RMSE": tensor(0.1), ...}

# Compute all defaults
calc_all = MetricCalculator(task="prediction", mode="probabilistic")
results_all = calc_all.compute_all(target, samples)

# Per-feature breakdown
per_feat = calc.compute_per_feature(target, forecast)
# -> {"MSE": tensor([...]), "MAE": tensor([...])}
```

## 可用指标一览

### Prediction（预测）

| Mode | 指标 | 说明 |
|------|------|------|
| point | MSE, RMSE, MAE, MAPE, sMAPE, ND, R2, Correlation | 点预测指标 |
| probabilistic | CRPS, CRPS_sum, PICP, QICE, MSE_median, MAE_median, Calibration, LogLikelihood | 概率预测指标 |

### Imputation（插补）

| Mode | 指标 | 说明 |
|------|------|------|
| point | MSE, RMSE, MAE, MAPE, MRE, sMAPE, ND | 点插补指标 |
| probabilistic | CRPS, PICP, QICE, IntervalWidth | 概率插补指标 |

### Generation（生成）

| Mode | 指标 | 说明 |
|------|------|------|
| point | Fidelity, DiscriminativeScore, Correlation, KLDivergence | 样本级生成指标 |
| probabilistic | MMD, JSDivergence, LogLikelihood | 分布级生成指标 |

## 运行测试

```bash
cd ts_metric
pytest tests/ -v
```
