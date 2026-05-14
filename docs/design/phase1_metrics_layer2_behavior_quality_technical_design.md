# Phase I Metrics Layer 2 Technical Design: Behavior Quality

Layer 2 判断每个 code 是否对应可解释、相对稳定、彼此有区分度的交易行为。该层关注
行为结构，不直接评估收益；但可以消费 Layer 3 的 per-code profitability，用于判断
weak lift code 是否仍有保留价值。

## 1. 工程位置

计算代码建议放在：

```text
src/phase1/evaluators/phase1_validation_layers/layer2_behavior_quality.py
```

Layer 2 calculator 负责 morphology/motif 统计、code-level 行为诊断和 code 间
分离度。PASS/FAIL 由 `evaluate_behavior_quality_rules()` 判定。

## 2. 输入依赖

必需输入：

- `train_snapshot: Phase1EvaluationSnapshot`
- `val_snapshot: Phase1EvaluationSnapshot`
- `runtime_config: Phase1ValidationRuntimeConfig`
- `per_code_profitability: Sequence[Phase1PerCodeProfitability] | None`

实际使用字段：

- `val_snapshot.prices`
- `val_snapshot.decoded_actions`
- `val_snapshot.code_ids`
- `val_snapshot.z_e`
- Layer 3 输出的 `per_code_profitability`

Layer 2 的 code-level profitability 字段来自 Layer 3，不在本层重新执行交易收益。

## 3. 输出 Metrics

```python
@dataclass(frozen=True)
class Phase1BehaviorQualityMetrics:
    weak_support_code_ratio: float
    weak_morphology_code_ratio: float
    weak_motif_code_ratio: float
    weak_pair_code_ratio: float
    weak_lift_nonprofitable_code_ratio: float
    intra_code_action_similarity: float
    inter_intra_separation: float
    latent_silhouette_score: float
    duplicate_code_pair_count: int
    profitable_code_coverage: float
```

Code diagnostics：

```python
@dataclass(frozen=True)
class Phase1CodeDiagnostic:
    code_id: int
    support: int
    occupancy: float
    dominant_morphology: str | None
    dominant_morphology_ratio: float | None
    morphology_lift: float | None
    dominant_motif: str | None
    dominant_motif_ratio: float | None
    dominant_pair: str | None
    dominant_pair_ratio: float | None
    decoded_mean_advantage: float | None
    decoded_win_rate: float | None
    retention_ratio: float | None
    fee_drag: float | None
    status: str
```

## 4. 推荐入口

```python
def compute_behavior_quality_metrics(
    *,
    train_snapshot: Phase1EvaluationSnapshot,
    val_snapshot: Phase1EvaluationSnapshot,
    runtime_config: Phase1ValidationRuntimeConfig,
    per_code_profitability: Sequence[Phase1PerCodeProfitability] | None = None,
) -> Phase1LayerComputation:
    ...
```

## 5. Helper 设计

```python
def classify_market_morphology(prices: np.ndarray) -> np.ndarray:
    ...

def classify_action_motif(
    actions: np.ndarray,
    prices: np.ndarray | None,
) -> np.ndarray:
    ...

def compute_distribution_by_code(
    values: np.ndarray,
    code_ids: np.ndarray,
) -> dict[int, dict[str, float]]:
    ...

def compute_lift(
    code_distribution: Mapping[str, float],
    global_distribution: Mapping[str, float],
) -> dict[str, float]:
    ...

def compute_intra_code_action_similarity(
    actions: np.ndarray,
    code_ids: np.ndarray,
) -> float:
    ...

def compute_inter_intra_separation(
    actions: np.ndarray,
    code_ids: np.ndarray,
) -> float:
    ...

def compute_duplicate_code_pair_count(
    code_prototypes: np.ndarray,
    threshold: float,
) -> int:
    ...
```

## 6. 市场形态分类

对每个 validation horizon 的价格序列 `p_0, ..., p_{h-1}`，先计算：

- `ret_total = p_{h-1} / p_0 - 1`
- `ret_first = p_mid / p_0 - 1`
- `ret_second = p_{h-1} / p_mid - 1`
- `realized_vol = std(log(p_t / p_{t-1})) * sqrt(h)`
- `max_drawdown`
- `max_runup`
- `range_ratio = (max(p) - min(p)) / p_0`
- `trend_efficiency = abs(p_{h-1} - p_0) / sum_t abs(p_t - p_{t-1})`

阈值建议用 validation set 的分位数自适应确定：

- `vol_high`: `realized_vol` 的 70% 分位数；
- `vol_low`: `realized_vol` 的 30% 分位数；
- `range_high`: `range_ratio` 的 70% 分位数；
- `trend_ret_threshold`: `abs(ret_total)` 的 60% 分位数，且至少大于单边交易成本的 3 倍；
- `reversal_leg_threshold`: `abs(ret_first)` 和 `abs(ret_second)` 合并后的 60% 分位数，
  且至少大于单边交易成本的 2 倍。

类别：

- `uptrend`
- `downtrend`
- `reversal-up`
- `reversal-down`
- `range-high-vol`
- `range-low-vol`
- `volatile-mixed`
- `neutral`

归类优先级：

1. `reversal-up` / `reversal-down`
2. `uptrend` / `downtrend`
3. `range-high-vol` / `range-low-vol`
4. `volatile-mixed`
5. `neutral`

## 7. 行为 Motif 分类

motif 把 decoded action sequence 映射为粗粒度交易意图。第一版建议由以下维度组合：

- 主方向：`long`、`short`、`flat`、`mixed`
- 入场时点：`early`、`middle`、`late`、`none`
- 持仓长度：`hold`、`short-hold`、`mostly-flat`
- 与近期价格方向关系：`with-recent-move`、`against-recent-move`、`unknown`

示例 motif：

- `long+early+hold+with-recent-move`
- `short+middle+short-hold+against-recent-move`
- `flat+none+mostly-flat+unknown`

motif 的目标不是还原每个 timestep，而是让同一 code 内的行为一致性可以统计。

## 8. 计算流程

1. 对每个 validation horizon 用价格序列分类 morphology。
2. 对每条 decoded action sequence 分类 motif。
3. 对每个 active code 统计 support、occupancy。
4. 统计 `P(morphology | code)`，得到 dominant morphology 和 ratio。
5. 统计 `P(motif | code)`，得到 dominant motif 和 ratio。
6. 统计 `P(morphology, motif | code)`，得到 dominant pair 和 ratio。
7. 用全体验证集 `P(morphology)` 计算 dominant morphology lift。
8. 统计 support 低于 `max(100, 0.02 * N_val)` 的 active code 比例，得到
   `weak_support_code_ratio`。
9. 统计 dominant morphology ratio、motif ratio、pair ratio 不达标的 code 比例。
10. 结合 Layer 3 的 per-code profitability，统计 morphology lift 不足且不盈利的
    code 比例，得到 `weak_lift_nonprofitable_code_ratio`。
11. 计算 intra-code action similarity。第一版可用逐 timestep position 一致率：
    对同一 code 内样本两两比较或抽样比较，求平均相似度。
12. 计算 inter/intra separation。第一版可将每个 code 的 decoded action 转为
    position sequence 均值原型，计算 code 间中心距离 / code 内平均距离。
13. 计算 latent silhouette score。若 active code 少于 2，写入 `nan`。
14. 计算 duplicate code pair count。任意两个 code 原型相似度超过阈值即计数。
15. 计算 `profitable_code_coverage`：Layer 3 中 per-code 盈利条件通过的 active
    code 数量 / active code 数量。

## 9. Rule 判定

Layer 2 thresholds：

```python
@dataclass(frozen=True)
class Phase1BehaviorQualityThresholds:
    min_code_support_abs: int = 100
    min_code_support_ratio: float = 0.02
    weak_code_ratio_max: float = 0.20
    dominant_morphology_ratio_min: float = 0.35
    dominant_motif_ratio_min: float = 0.40
    dominant_pair_ratio_min: float = 0.30
    morphology_lift_min: float = 1.25
    intra_code_similarity_min: float = 0.65
    inter_intra_separation_min: float = 1.30
    duplicate_code_similarity_max: float = 0.85
```

Hard gates：

- `weak_support_code_ratio <= 0.20`
- `weak_morphology_code_ratio <= 0.20`
- `weak_motif_code_ratio <= 0.20`
- `weak_pair_code_ratio <= 0.20`
- `weak_lift_nonprofitable_code_ratio <= 0.20`
- `intra_code_action_similarity >= 0.65`
- `inter_intra_separation >= 1.30`
- `duplicate_code_pair_count == 0`
- `profitable_code_coverage >= 0.60`

`latent_silhouette_score` 第一版可作为 warn/scoring 信号；active code 少于 2 时
应直接 fail，因为 code 间分离度不可定义。

## 10. 缺失数据策略

- 缺少 `prices` 时，morphology 和 against/with recent move motif 不可靠；
  morphology 相关 hard gate 应 fail；
- 如果 Layer 3 尚未完成，`profitable_code_coverage` 和
  `weak_lift_nonprofitable_code_ratio` 写入 `nan`，正式 selector 不应使用该
  checkpoint；
- active code 少于 2 时，inter/intra separation 和 silhouette 不可计算，应 fail；
- 单个 code support 太低时，该 code 的 dominant ratio 不应被当作强证据。

## 11. 与其他层的关系

- Layer 1 提供稳定 code assignment 的前提；
- Layer 3 的 per-code profitability 会补全 `Phase1CodeDiagnostic` 的收益字段；
- Layer 4 的 conditional entropy 或 MI 计算可以复用 morphology label，但不能使用
 未来价格路径作为 selector probe 输入；
- report 的 code-level 表格主要来自 Layer 2 diagnostics。

## 12. 测试要点

- morphology 分类在趋势、反转、横盘、高波动样本上稳定；
- motif 分类对 flat-only、single-entry、multi-switch 序列有明确输出；
- lift 计算对全局分布为 0 的类别有 eps 保护；
- duplicate code 计数不会重复计同一对 code；
- per-code profitability 缺失时 formal selection fail，而不是静默通过。
