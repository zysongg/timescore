# Prediction Metrics

## Point Metrics (回归指标)

### MSE (Mean Squared Error)

$$\text{MSE} = \frac{1}{B \cdot C \cdot T} \sum_{b,c,t} (y_{b,c,t} - \hat{y}_{b,c,t})^2$$

- **含义**: 预测值与真实值之间的均方误差
- **范围**: $[0, +\infty)$
- **越小越好**

### RMSE (Root Mean Squared Error)

$$\text{RMSE} = \sqrt{\text{MSE}}$$

- **含义**: MSE 的平方根，与原始数据量纲一致
- **越小越好**

### NRMSE (Normalized RMSE) — GluonTS 兼容

$$\text{NRMSE} = \frac{\text{RMSE}}{\text{mean}(|y|)}$$

- **含义**: 归一化 RMSE，除以真实值绝对值的均值
- **越小越好**
- **参考**: GluonTS Evaluator

### MAE (Mean Absolute Error)

$$\text{MAE} = \frac{1}{B \cdot C \cdot T} \sum_{b,c,t} |y_{b,c,t} - \hat{y}_{b,c,t}|$$

- **含义**: 预测值与真实值之间的平均绝对误差
- **越小越好**

### MAPE (Mean Absolute Percentage Error)

$$\text{MAPE} = \frac{1}{N} \sum_{(b,c,t) \in \mathcal{V}} \frac{|y_{b,c,t} - \hat{y}_{b,c,t}|}{|y_{b,c,t}|}$$

其中 $\mathcal{V}$ 是 $|y| > \epsilon$ 的有效位置集合，$N = |\mathcal{V}|$。

- **含义**: 相对误差的均值（百分比形式）
- **范围**: $[0, +\infty)$
- **越小越好**
- **注意**: 当真实值接近 0 时不稳定

### sMAPE (Symmetric MAPE)

$$\text{sMAPE} = \frac{1}{N} \sum_{(b,c,t) \in \mathcal{V}} \frac{2 |y_{b,c,t} - \hat{y}_{b,c,t}|}{|y_{b,c,t}| + |\hat{y}_{b,c,t}|}$$

- **含义**: 对称版 MAPE，分母同时考虑真实值和预测值
- **范围**: $[0, 2]$
- **越小越好**
- **优势**: 比 MAPE 更稳定，不会因预测值为 0 而爆炸

### ND (Normalized Deviation)

$$\text{ND} = \frac{\sum_{b,c,t} |y_{b,c,t} - \hat{y}_{b,c,t}|}{\sum_{b,c,t} |y_{b,c,t}|}$$

- **含义**: 归一化偏差，总绝对误差除以总绝对真实值
- **越小越好**
- **常用于**: 时序预测基准（如 DeepAR 论文）

### R² (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum_{b,c,t} (y_{b,c,t} - \hat{y}_{b,c,t})^2}{\sum_{b,c,t} (y_{b,c,t} - \bar{y})^2}$$

- **含义**: 决定系数，衡量模型解释方差的能力
- **范围**: $(-\infty, 1]$，通常 $[0, 1]$
- **越大越好**，$R^2 = 1$ 表示完美预测

### Correlation (Pearson)

$$\text{Corr} = \frac{1}{B \cdot C} \sum_{b,c} \text{corr}(y_{b,c,:}, \hat{y}_{b,c,:})$$

- **含义**: 每个样本-特征对的 Pearson 相关系数均值
- **范围**: $[-1, 1]$
- **越大越好**，1 表示完美线性相关

---

## Probabilistic Metrics (概率指标)

### CRPS (Continuous Ranked Probability Score)

$$\text{CRPS} = \mathbb{E}|X - y| - \frac{1}{2} \mathbb{E}|X - X'|$$

其中 $X, X'$ 是从预测分布中独立抽取的样本。对于 $S$ 个集成样本的精确计算：

$$\text{CRPS} = \frac{1}{S} \sum_{s=1}^{S} |x_s - y| - \frac{1}{S^2} \sum_{s=1}^{S} (2s - 1 - S) \cdot x_{(s)}$$

其中 $x_{(s)}$ 是排序后的第 $s$ 个样本。

- **含义**: 连续排序概率得分，衡量预测分布与真实值的整体匹配度
- **越小越好**，0 表示完美预测
- **实现**: `crps()` 使用精确公式，$O(S \log S)$ 复杂度（通过排序优化）
- **验证**: 与 `CRPS.CRPS` 包（NsDiff 参考实现）结果完全一致

### CRPS_quantile (分位数近似)

$$\text{CRPS\_quantile} \approx \frac{2}{Q} \sum_{q \in \mathcal{Q}} \text{QL}(q, y, \hat{y}^{(q)})$$

其中 $\mathcal{Q} = \{0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95\}$ 是分位数集合，$\text{QL}$ 是分位数损失：

$$\text{QL}(q, y, \hat{y}) = \max(q \cdot (y - \hat{y}), (q-1) \cdot (y - \hat{y}))$$

- **含义**: CRPS 的分位数损失近似（GluonTS/DeepAR 风格）
- **越小越好**
- **精度**: 约 4-5% 误差（受 7 个分位点限制），速度更快

### CRPS_sum

$$\text{CRPS\_sum} = \text{CRPS}\left(\sum_c y_{b,c,t}, \sum_c s_{b,s,c,t}\right)$$

- **含义**: 对所有特征求和后的 CRPS，衡量边际分布质量
- **越小越好**

---

### GluonTS 兼容概率指标

以下指标的公式与 GluonTS Evaluator 完全一致，已通过数值验证。

#### QuantileLoss[q] — 分位数损失

$$\text{QL}(q) = 2 \sum_{b,c,t} \left|(\hat{y}^{(q)} - y) \cdot (\mathbb{1}[y \leq \hat{y}^{(q)}] - q)\right|$$

默认分位点: $q \in \{0.1, 0.2, \ldots, 0.9\}$

- **含义**: 每个分位点水平的加权 pinball loss（sum 而非 mean）
- **越小越好**
- **返回**: `dict` 映射 $q$ → 标量

#### wQuantileLoss[q] — 加权分位数损失

$$\text{wQL}(q) = \frac{\text{QL}(q)}{\sum |y|}$$

- **含义**: QuantileLoss 除以真实值绝对值之和，实现跨序列可比
- **越小越好**

#### mean_wQuantileLoss

$$\text{mean\_wQuantileLoss} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \text{wQL}(q)$$

- **含义**: 所有分位点的 wQuantileLoss 均值，GluonTS 的核心汇总指标
- **越小越好**

#### mean_absolute_QuantileLoss

$$\text{mean\_absolute\_QuantileLoss} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \text{QL}(q)$$

- **含义**: 所有分位点的 QuantileLoss 均值
- **越小越好**

#### Coverage[q] — 覆盖率

$$\text{Coverage}(q) = \frac{1}{B \cdot C \cdot T} \sum_{b,c,t} \mathbb{1}[y_{b,c,t} \leq \hat{y}^{(q)}_{b,c,t}]$$

- **含义**: 真实值低于分位数预测的比例
- **范围**: $[0, 1]$
- **理想值**: $q$（如 $q=0.5$ 时理想覆盖率为 0.5）
- **返回**: `dict` 映射 $q$ → 标量

#### MAE_Coverage

$$\text{MAE\_Coverage} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} |\text{Coverage}(q) - q|$$

- **含义**: 覆盖率与期望覆盖率的平均绝对偏差，衡量校准质量
- **越小越好**，0 表示完美校准

#### MSIS (Mean Scaled Interval Score) — M4 竞赛指标

$$\text{MSIS} = \frac{1}{\text{seasonal\_error}} \cdot \text{mean}\left(U - L + \frac{2}{\alpha}(L - y)\mathbb{1}[y < L] + \frac{2}{\alpha}(y - U)\mathbb{1}[y > U]\right)$$

其中 $L = \hat{y}^{(\alpha/2)}$, $U = \hat{y}^{(1-\alpha/2)}$，默认 $\alpha = 0.05$（95% 区间）。

`seasonal_error` 默认由 `mean(|y[t] - y[t-1]|)` 估计，也可通过参数传入。

- **含义**: 缩放区间得分，同时衡量区间宽度和覆盖率
- **越小越好**
- **参考**: M4 Competition, GluonTS

### PICP (Prediction Interval Coverage Probability)

$$\text{PICP} = \frac{1}{B \cdot C \cdot T} \sum_{b,c,t} \mathbb{1}\left[\hat{y}^{(\alpha/2)}_{b,c,t} \leq y_{b,c,t} \leq \hat{y}^{(1-\alpha/2)}_{b,c,t}\right]$$

- **含义**: 预测区间覆盖率，衡量置信区间的校准程度
- **范围**: $[0, 1]$
- **理想值**: $1 - \alpha$（如 $\alpha=0.1$ 时理想值为 0.9）

### QICE (Quantile Interval Coverage Error)

$$\text{QICE} = \frac{1}{L} \sum_{\alpha \in \mathcal{A}} |\text{PICP}(\alpha) - (1 - \alpha)|$$

其中 $\mathcal{A} = \{0.1, 0.2, \ldots, 0.9\}$，$L = |\mathcal{A}|$。

- **含义**: 分位数区间覆盖误差，衡量多个置信水平的平均校准偏差
- **越小越好**，0 表示完美校准

### MSE_median / MAE_median

$$\text{MSE\_median} = \frac{1}{B \cdot C \cdot T} \sum_{b,c,t} (y_{b,c,t} - \tilde{y}_{b,c,t})^2$$

其中 $\tilde{y}_{b,c,t}$ 是样本的中位数（50% 分位数）。

- **含义**: 中位数预测的 MSE/MAE
- **越小越好**

### Calibration Error

$$\text{Calibration} = \frac{1}{L} \sum_{q \in \mathcal{Q}} \left| \frac{1}{B \cdot C \cdot T} \sum_{b,c,t} \mathbb{1}[y_{b,c,t} \leq \hat{y}^{(q)}_{b,c,t}] - q \right|$$

- **含义**: 校准误差，衡量预测分位数与实际覆盖率的偏差
- **越小越好**

### Log-Likelihood

$$\text{LL} = \frac{1}{B \cdot C \cdot T} \sum_{b,c,t} -\frac{1}{2} \left( \log(2\pi \sigma^2_{b,c,t}) + \frac{(y_{b,c,t} - \mu_{b,c,t})^2}{\sigma^2_{b,c,t}} \right)$$

其中 $\mu$ 和 $\sigma^2$ 是样本的均值和方差。

- **含义**: 高斯假设下的对数似然
- **越大越好**

---

## Mask 支持

所有指标支持可选 `mask` 参数（shape: `(B, C, T)` 或可广播）：
- `mask=1` 的位置参与计算
- `mask=0` 的位置被忽略
- 自动归一化：$\text{MSE} = \frac{\sum m \cdot (y - \hat{y})^2}{\sum m}$
