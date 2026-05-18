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
- `val_snapshot.demo_rewards`
- `val_snapshot.decoded_actions`
- `val_snapshot.code_ids`
- `runtime_config.fee_rate`
- `runtime_config.random_label_trials`
- `runtime_config.random_seed`
- `runtime_config.top_contribution_ratio`
- `runtime_config.active_code_min_occupancy`

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
    random_label_risk_adjusted_return: float = float("nan")
    risk_adjusted_return_vs_random: float = float("nan")
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
    thresholds: Phase1OracleProfitabilityThresholds | None = None,
) -> Phase1LayerComputation:
    ...
```

## 5. 统一 Execution 口径

输入：

```python
prices: np.ndarray | None  # [n, h] or [n, h, 1]
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
    prices: np.ndarray | None,
    actions: np.ndarray,
    fee_rate: float,
) -> ExecutionResult:
    ...

def decode_labels(
    *,
    model: ArchetypeVQModel,
    states: np.ndarray,
    code_ids: np.ndarray,
    device: torch.device | str,
) -> np.ndarray:
    ...

def decode_random_labels(
    *,
    model: ArchetypeVQModel,
    states: np.ndarray,
    num_archetypes: int,
    trials: int,
    seed: int,
    device: torch.device | str,
) -> np.ndarray:
    ...

def compute_random_label_returns(
    *,
    model: ArchetypeVQModel,
    snapshot: Phase1EvaluationSnapshot,
    runtime_config: Phase1ValidationRuntimeConfig,
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
    *,
    code_ids: np.ndarray,
    decoded_advantage: np.ndarray,
    decoded_returns: np.ndarray,
    dp_advantage: np.ndarray,
    decoded_gross_returns: np.ndarray,
    decoded_fees: np.ndarray,
    thresholds: Phase1OracleProfitabilityThresholds,
    active_codes: Sequence[int] | None = None,
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

当前实现中，execution 底层口径收敛到
`src/utils/trade_execution.py::ActionExecutionCalculator.execute_actions()`，Layer 3
通过 `layer3_oracle_profitability.py::execute_actions()` 暴露公共 wrapper。Layer 4
的 probe label decode 和 return retention 复用 Layer 3 的 `decode_labels()` 与
`execute_actions()`。旧 `_xxx` 私有 helper 名称仅作为兼容别名保留，新代码应使用
本节列出的公共 API。

各 helper 的功能定义如下。

### 6.1 `execute_actions`

该 helper 是 Layer 3 最重要的共享口径，负责把价格路径和动作序列转换成逐 horizon
收益。

输入：

- `prices`：价格数组，支持 `[N, H]` 或 `[N, H, 1]`；
- `actions`：动作数组，形状 `[N, H]`，动作值为 `0/1/2`；
- `fee_rate`：单边换手手续费率。

输出：

- `returns`：`gross_returns - fees`，即扣除手续费后的净收益；
- `gross_returns`：动作持仓作用在 `price_t -> price_{t+1}` 上得到的未扣费收益；
- `fees`：按全 horizon 换手量累计的手续费；
- `turnover`：从初始 flat position 开始累计的持仓变化绝对值。

执行细节：

- 动作映射固定为 `0=short`、`1=flat`、`2=long`，对应持仓 `-1/0/1`；
- `position_t` 只作用于 `price_t -> price_{t+1}`，最后一个 action 不贡献收益；
- 初始持仓按 flat 计算，因此第一步从 flat 切到 short/long 会产生换手；
- `prices` 缺失、维度不合法或 horizon 小于 2 时，返回同样 sample 数的 `NaN`
  结果，而不是抛出收益类异常；
- `prices` 与 `actions` 的 sample 数不一致时应抛出 `ValueError`。

该 helper 必须被 Layer 3 的 decoded returns、random label baseline、Layer 4 的
`probe_return_retention` 以及需要统一收益口径的 Layer 0 逻辑复用。report 和
selector 不允许重新实现该口径。

### 6.2 `decode_random_labels`

该 helper 用于构造 random label baseline，判断 assigned code label 是否真的携带交易
信息，而不是 decoder 对任意 label 都能产生类似收益。

输入：

- `model`：包含 `quantizer.embedding_from_code()` 和 `decoder()` 的 VQ 模型；
- `states`：验证集状态序列，形状 `[N, H, state_dim]`；
- `num_archetypes`：可采样 code 数；
- `trials`：随机 label 采样次数；
- `seed`：随机数种子，必须写入 payload/report；
- `device`：decoder 推理设备。

计算步骤：

1. 用 `seed` 初始化 deterministic RNG；
2. 每个 trial 为每个样本从 `[0, num_archetypes)` 均匀采样一个 random label；
3. 调用 `model.quantizer.embedding_from_code(random_labels)` 得到 label embedding；
4. 调用 frozen decoder 生成 action logits，并取 `argmax` 得到 random actions；
5. 使用 `execute_actions()` 执行 random actions；
6. 多个 trial 的 random returns 按样本求均值，得到 `R_rand`。

当前公共 API 拆成三层：

- `decode_labels()`：给定任意指定 code label，返回 `[N, H]` decoded actions；
- `decode_random_labels()`：多次采样 random label，返回 `[trials, N, H]`
  random-label decoded actions；
- `compute_random_label_returns()`：对 random-label actions 逐 trial 调用
  `execute_actions()`，返回 `[N]` 平均 random returns。

调用方如果只需要收益，应使用 `compute_random_label_returns()`；如果需要审计 random
actions，则使用 `decode_random_labels()` 后仍必须走本层 `execute_actions()`。

### 6.3 `compute_max_drawdown`

该 helper 计算累计收益序列的最大回撤，用于衡量 decoded 策略自身的下行风险。

输入为一维 `cumulative_returns`。helper 先过滤非有限值，再计算：

```text
peak_t = max(cumulative_returns_0 ... cumulative_returns_t)
drawdown_t = peak_t - cumulative_returns_t
max_drawdown = max(drawdown_t)
```

无有效样本时返回 `NaN`。Layer 3 使用方式为：

```text
downside_control = max_drawdown(cumsum(R_dec))
```

`downside_control` 越低越好；它是 decoded 策略累计收益曲线的绝对最大回撤。
DP teacher 的累计收益路径可能按构造长期无回撤，因此不能作为该指标的分母。

### 6.4 `compute_risk_adjusted_return`

该 helper 计算跨 validation horizons 的轻量风险调整收益：

```text
risk_adjusted_return = mean(returns) / (std(returns) + eps)
```

输入为逐样本收益数组，helper 应过滤 `NaN/inf`。无有效样本时返回 `NaN`。该指标不是
年化 Sharpe，也不引入交易频率假设，只用于比较同一 validation snapshot 内 decoded
收益的均值与离散程度。

Layer 3 至少应计算：

- assigned label decoded returns 的 `risk_adjusted_return`；
- random label returns 的 `random_label_risk_adjusted_return`；
- 二者差值 `risk_adjusted_return_vs_random`。

这样可以避免模型只在均值上略优于 random baseline，但风险调整后更差。

### 6.5 `compute_top_contribution_ratio`

该 helper 检测盈利是否过度依赖少数 extreme horizon。

输入：

- `returns`：通常传 `decoded_advantage` 或 decoded profit；
- `top_ratio`：头部样本比例，例如 `0.05`。

推荐口径：

1. 过滤非有限值；
2. 取正收益样本作为 denominator；
3. 以全部有效样本数乘以 `top_ratio` 得到 top 样本数量，至少取 1 个；
4. 取收益最高的 top 样本；
5. 返回 `sum(top_returns) / (sum(positive_returns) + eps)`。

如果没有正收益，返回 `NaN`。该指标越高，说明收益越集中；Layer 3 hard gate 默认要求
`top_5_contribution <= top_5_contribution_max`。

### 6.6 `compute_per_code_profitability`

该 helper 把全局收益拆成 per-code profitability 摘要，是 Layer 3 与 Layer 2 的主要
连接点。

输入：

- `code_ids`：每个样本的 assigned code；
- `decoded_advantage`：`R_dec - R_flat`；
- `decoded_returns`：decoded 策略净收益；
- `dp_advantage`：`R_DP - R_flat`；
- `decoded_gross_returns`：decoded 策略 gross returns；
- `decoded_fees`：decoded 策略手续费；
- `thresholds`：per-code 判定阈值；
- `active_codes`：可选，只统计 occupancy 达标的 code。

每个 active code 计算：

```text
mean_advantage = mean(decoded_advantage[code])
win_rate = mean(decoded_returns[code] > 0)
retention_ratio = sum(decoded_advantage[code]) / (sum(dp_advantage[code]) + eps)
fee_drag = sum(decoded_fees[code]) / (sum(positive decoded_gross_returns[code]) + eps)
```

其中 per-code `retention_ratio` 与全局 retention 使用同一安全分母口径：
`sum(dp_advantage[code]) <= eps` 或不可计算时返回 `NaN`。

`passed` 字段推荐按以下条件生成：

```text
mean_advantage > 0
and win_rate >= per_code_win_rate_min
and retention_ratio >= per_code_retention_ratio_min
and fee_drag <= per_code_fee_drag_max
```

Layer 3 使用 per-code 结果计算 `bad_code_ratio`。Layer 2 可以复用同一 payload 计算
profitable-code coverage、weak-lift code 等行为质量指标，避免重复定义盈利 code。

### 6.7 其他内部 helper

实现中还可以包含以下私有 helper，但它们不需要成为稳定公共接口：

- `demo_returns` helper：优先使用 `demo_rewards.sum(axis=1)` 作为 DP teacher return，
  同时用 `execute_actions()` 补齐 gross returns、fees 和 turnover；当
  `demo_rewards` 缺失或不可信时，退回统一执行口径。
- `active_codes` helper：按 `runtime_config.active_code_min_occupancy` 过滤 active
  code，避免低占用 code 对 per-code 统计和 `bad_code_ratio` 造成噪声。
- `trimmed_mean` helper：对 `decoded_advantage` 做双侧截尾均值，用于
  `trimmed_decoded_advantage`。
- `safe_corr` helper：计算 `turnover` 与 `R_dec` 的 Pearson correlation；有效样本不足
  时返回 `NaN`，任一侧方差接近 0 时返回 0。
- `fee_drag` helper：统一计算 `total_fee / gross_profit`；当 gross profit 不为正时返回
  `inf`，使 rules 层能够明确失败。
- `dominant_pair_positive_ratio` helper：结合 Layer 2 的 market morphology 与 action
  motif，统计 active code 的 dominant pair 中 mean decoded advantage 为正的比例。
- `safe_retention_ratio` helper：当 `sum(dp_advantage)` 缺失、非正或接近 0 时返回
  `NaN`，避免 teacher 无盈利能力时 retention ratio 被误解释为有效指标。
- `missing_prices_computation` helper：当 `prices` 缺失或无效时直接返回全 `NaN`
  metrics 与 NaN returns payload，使 rules 层按 skip-as-fail 处理。

## 7. 计算流程

1. 检查 `prices` 是否可用于执行收益计算。缺失、维度不合法、horizon 小于 2 或包含
   非有限值时，直接返回全 `NaN` metrics。
2. 用 execution helper 计算 `R_DP`、`R_dec`、`R_flat`。
3. 用随机 label decode 得到 `random_actions`，执行后得到 `R_rand`。多次 trial
   时对 random returns 取均值。
4. `decoded_advantage = R_dec - R_flat`。
5. `dp_advantage = R_DP - R_flat`。
6. `mean_decoded_advantage_vs_flat = mean(decoded_advantage)`。
7. `decoded_win_rate_vs_flat = mean(R_dec > R_flat)`。
8. `mean_advantage_vs_random_label = mean(R_dec - R_rand)`。
9. `random_label_relative_lift = mean(R_dec - R_rand) / (abs(mean(R_rand - R_flat)) + eps)`。
10. `retention_ratio = sum(decoded_advantage) / sum(dp_advantage)`；当
    `sum(dp_advantage) <= eps` 或不可计算时返回 `NaN`。
11. `downside_control = max_drawdown(cumsum(R_dec))`。
12. `risk_adjusted_return = mean(R_dec) / (std(R_dec) + eps)`。
13. `random_label_risk_adjusted_return = mean(R_rand) / (std(R_rand) + eps)`。
14. `risk_adjusted_return_vs_random = risk_adjusted_return - random_label_risk_adjusted_return`。
15. `top_5_contribution`：取 decoded profit 为正的样本，计算收益最高 top 5%
    对总正收益的贡献。
16. `trimmed_decoded_advantage`：去掉 decoded advantage 最高和最低各
    `runtime_config.top_contribution_ratio` 后求均值。
17. `fee_drag = total_fee / (gross_profit + eps)`；当 gross profit 不为正时返回 `inf`。
18. `turnover_return_correlation = corr(turnover, R_dec)`。
19. 按 active code 统计 per-code mean advantage、win rate、retention、fee drag。
20. 计算 `bad_code_ratio`：per-code mean advantage 小于 0 的 active code 比例。
21. 结合 Layer 2 的 dominant pair 或本层临时 pair 统计，计算
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
    downside_control_max: float = 2.00
    risk_adjusted_return_min: float = 0.0
    fee_drag_max: float = 0.35
    turnover_return_correlation_min: float = -0.10
    per_code_win_rate_min: float = 0.52
    per_code_retention_ratio_min: float = 0.40
    per_code_fee_drag_max: float = 0.40
```

Hard gates：

- `mean_decoded_advantage_vs_flat > 0`
- `decoded_win_rate_vs_flat >= 0.55`
- `mean_advantage_vs_random_label > 0`
- `random_label_relative_lift >= 0.20`
- `retention_ratio >= 0.50`
- `downside_control <= 2.00`
- `risk_adjusted_return > 0`
- `risk_adjusted_return_vs_random > 0`
- `top_5_contribution <= 0.60`
- `trimmed_decoded_advantage > 0`
- `fee_drag <= 0.35`
- `turnover_return_correlation >= -0.10`
- `bad_code_ratio <= 0.30`
- `dominant_pair_positive_ratio >= 0.60`

当前实现中上述指标均为 hard gate。任一 hard gate 缺失或为 `NaN` 时，rules 层通过
skip-as-fail 机制将本层判为失败。`risk_adjusted_return`、`retention_ratio` 等仍会进入
score 和 tie-breaker，用于已通过 checkpoint 之间排序。

## 9. 缺失数据策略

- 缺少 `prices`、价格维度不合法、horizon 小于 2 或价格包含非有限值时，本层返回全
  `NaN` metrics，由 rules 层 skip-as-fail；
- `sum(dp_advantage) <= eps` 时 retention ratio 不可靠，写入 `NaN`，并应同时反映
  Layer 0 teacher quality 失败；
- `max_drawdown(cumsum(R_DP)) <= eps` 时 downside control 分母不可靠，写入 `NaN`；
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
- 缺失价格时所有 hard gate metrics 均为 `NaN` 并由 rules 层 skip-as-fail；
- global/per-code retention 的 DP 分母接近 0 时写入 `NaN` 且不会误判通过。
