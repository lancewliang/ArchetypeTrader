# Phase I Metrics Layer 3 Technical Design: Oracle Profitability

Layer 3 判断 oracle assigned-label 经过 frozen decoder 执行后，是否仍保留 DP
teacher 的盈利能力。该层是 Phase I codebook 是否有交易价值的核心验证。

## 1. 工程位置

计算代码建议放在：

```text
src/phase1/evaluators/phase1_validation_layers/layer3_oracle_profitability.py
```

Layer 3 统一负责收益执行口径、random label baseline、retention、fee drag 和
per-code profitability。report 和 selector 不允许重新实现收益计算。

## 2. 输入依赖

必需输入：

- `model: ArchetypeVQModel`
- `val_snapshot: Phase1EvaluationSnapshot`
- `runtime_config: Phase1ValidationRuntimeConfig`
- `device: torch.device | str`

实际使用字段：

- `val_snapshot.states`
- `val_snapshot.prices`
- `val_snapshot.demo_actions`
- `val_snapshot.decoded_actions`
- `val_snapshot.code_ids`
- `runtime_config.fee_rate`
- `runtime_config.random_label_trials`
- `runtime_config.random_seed`
- `runtime_config.top_contribution_ratio`

Layer 3 需要 `model` 是因为 random label baseline 需要用随机 code 重新 decode。
如果上层 evaluator 已经预先收集 `random_label_decoded_actions`，入口可改为只接收
这些 actions，以减少本层对 model 的依赖。

## 3. 输出 Metrics

```python
@dataclass(frozen=True)
class Phase1OracleProfitabilityMetrics:
    mean_decoded_advantage_vs_flat: float
    decoded_win_rate_vs_flat: float
    mean_advantage_vs_random_label: float
    random_label_relative_lift: float
    retention_ratio: float
    downside_control: float
    risk_adjusted_return: float
    top_5_contribution: float
    trimmed_decoded_advantage: float
    fee_drag: float
    turnover_return_correlation: float
    bad_code_ratio: float
    dominant_pair_positive_ratio: float
```

Per-code profitability：

```python
@dataclass(frozen=True)
class Phase1PerCodeProfitability:
    code_id: int
    mean_advantage: float
    win_rate: float
    retention_ratio: float
    fee_drag: float
    passed: bool
```

返回 payload：

```python
Phase1LayerComputation(
    layer_id=3,
    layer_name="oracle_profitability",
    metrics=Phase1OracleProfitabilityMetrics(...),
    extra_payload={
        "per_code_profitability": tuple(...),
        "decoded_returns": ...,
        "dp_returns": ...,
        "flat_returns": ...,
        "random_label_returns": ...,
        "random_seed": runtime_config.random_seed,
    },
)
```

## 4. 推荐入口

```python
def compute_oracle_profitability_metrics(
    *,
    model: ArchetypeVQModel,
    val_snapshot: Phase1EvaluationSnapshot,
    runtime_config: Phase1ValidationRuntimeConfig,
    device: torch.device | str,
) -> Phase1LayerComputation:
    ...
```

## 5. 统一 Execution 口径

输入：

```python
prices: np.ndarray       # [n, h] or [n, h, 1]
actions: np.ndarray      # [n, h], values in {0, 1, 2}
fee_rate: float
```

动作映射：

```python
0 -> -1  # short
1 -> 0   # flat
2 -> 1   # long
```

收益定义：

```text
position_t = {-1, 0, 1}[action_t]
bar_return_t = price_{t+1} / price_t - 1
gross_return_t = position_t * bar_return_t
turnover_t = abs(position_t - position_{t-1})
fee_t = turnover_t * fee_rate
net_return_t = gross_return_t - fee_t
R_i = sum_t net_return_t
```

注意：

- horizon 最后一个 action 没有下一根 bar，可不计收益；
- 初始 position 默认为 flat；
- 若上游 DP planner 的 reward 定义不同，需要在 config 中记录并统一；
- report 中必须注明 execution 口径。

## 6. Helper 设计

```python
def execute_actions(
    prices: np.ndarray,
    actions: np.ndarray,
    fee_rate: float,
) -> ExecutionResult:
    ...

def decode_random_labels(
    model: ArchetypeVQModel,
    states: np.ndarray,
    num_archetypes: int,
    trials: int,
    seed: int,
    device: torch.device | str,
) -> np.ndarray:
    ...

def compute_max_drawdown(cumulative_returns: np.ndarray) -> float:
    ...

def compute_risk_adjusted_return(returns: np.ndarray) -> float:
    ...

def compute_top_contribution_ratio(
    returns: np.ndarray,
    top_ratio: float,
) -> float:
    ...

def compute_per_code_profitability(
    code_ids: np.ndarray,
    decoded_advantage: np.ndarray,
    dp_advantage: np.ndarray,
    fees: np.ndarray,
) -> tuple[Phase1PerCodeProfitability, ...]:
    ...
```

`ExecutionResult` 建议包含：

```python
@dataclass(frozen=True)
class ExecutionResult:
    returns: np.ndarray
    gross_returns: np.ndarray
    fees: np.ndarray
    turnover: np.ndarray
```

## 7. 计算流程

1. 用 execution helper 计算 `R_DP`、`R_dec`、`R_flat`。
2. 用随机 label decode 得到 `random_actions`，执行后得到 `R_rand`。多次 trial
   时对 random returns 取均值。
3. `decoded_advantage = R_dec - R_flat`。
4. `dp_advantage = R_DP - R_flat`。
5. `mean_decoded_advantage_vs_flat = mean(decoded_advantage)`。
6. `decoded_win_rate_vs_flat = mean(R_dec > R_flat)`。
7. `mean_advantage_vs_random_label = mean(R_dec - R_rand)`。
8. `random_label_relative_lift = mean(R_dec - R_rand) / (abs(mean(R_rand - R_flat)) + eps)`。
9. `retention_ratio = sum(decoded_advantage) / (sum(dp_advantage) + eps)`。
10. `downside_control = max_drawdown(cumsum(R_dec)) / (max_drawdown(cumsum(R_DP)) + eps)`。
11. `risk_adjusted_return = mean(R_dec) / (std(R_dec) + eps)`。
12. `top_5_contribution`：取 decoded profit 为正的样本，计算收益最高 top 5%
    对总正收益的贡献。
13. `trimmed_decoded_advantage`：去掉 decoded advantage 最高和最低各 5% 后求均值。
14. `fee_drag = total_fee / (gross_profit + eps)`。
15. `turnover_return_correlation = corr(turnover, R_dec)`。
16. 按 active code 统计 per-code mean advantage、win rate、retention、fee drag。
17. 计算 `bad_code_ratio`：per-code mean advantage 小于 0 的 active code 比例。
18. 结合 Layer 2 的 dominant pair 或本层临时 pair 统计，计算
    `dominant_pair_positive_ratio`。

## 8. Rule 判定

Layer 3 thresholds：

```python
@dataclass(frozen=True)
class Phase1OracleProfitabilityThresholds:
    decoded_win_rate_min: float = 0.55
    retention_ratio_min: float = 0.50
    random_label_relative_lift_min: float = 0.20
    bad_code_ratio_max: float = 0.30
    top_5_contribution_max: float = 0.60
    dominant_pair_positive_ratio_min: float = 0.60
```

Hard gates：

- `mean_decoded_advantage_vs_flat > 0`
- `decoded_win_rate_vs_flat >= 0.55`
- `random_label_relative_lift >= 0.20`
- `retention_ratio >= 0.50`
- `bad_code_ratio <= 0.30`
- `top_5_contribution <= 0.60`
- `trimmed_decoded_advantage > 0`
- `dominant_pair_positive_ratio >= 0.60`

`risk_adjusted_return` 进入 score 和 tie-breaker；`fee_drag`、`downside_control`、
`turnover_return_correlation` 第一版可作为 warning/diagnostic。

## 9. 缺失数据策略

- 缺少 `prices` 时，本层全部 hard gate 指标不可计算，应 fail；
- `sum(dp_advantage) <= 0` 时 retention ratio 不可靠，应同时反映 Layer 0 失败；
- `gross_profit <= 0` 时 fee drag 写入 `inf`；
- random baseline 必须固定 seed，并把 seed 写入 report payload；
- random decode 失败时，不能静默降级为 0 baseline，应标记本层 fail。

## 10. 与其他层的关系

- Layer 2 消费 `per_code_profitability`，用于 profitable-code coverage 和 weak-lift
  code 判断；
- Layer 4 的 `probe_return_retention` 必须复用本层 execution helper；
- tie-breaker 的第一优先级是 `risk_adjusted_return`；
- checkpoint selector 不重新执行收益，只读取 `validation.metrics` 和
  `validation.tie_breaker`。

## 11. 测试要点

- action 到 position 映射正确；
- 初始 flat position 的 turnover fee 正确；
- horizon 最后一个 action 不越界；
- random label baseline 在固定 seed 下可复现；
- top contribution 在全负收益或正收益为空时有定义；
- per-code retention 的分母接近 0 时有 eps 保护且不会误判通过。
