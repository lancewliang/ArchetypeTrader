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
    num_codes: int = 0
```

`num_codes` 表示当前 codebook size `K`，用于 rule 层在
`duplicate_code_pair_count_max is None` 时动态设置重复 code pair 上限。

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

`status` 是 per-code 综合诊断状态，由 support、morphology 清晰度、motif 清晰度、
morphology-motif pair 稳定性，以及 morphology lift / profitability 辅助证据共同
决定。推荐状态含义：

- `pass`：support 和结构/盈利辅助证据均达标；
- `weak`：存在少量结构弱项，但未达到严重不可靠；
- `bad`：support 不足，或多个结构弱项叠加。

## 4. 推荐入口

```python
def compute_behavior_quality_metrics(
    *,
    train_snapshot: Phase1EvaluationSnapshot,
    val_snapshot: Phase1EvaluationSnapshot,
    runtime_config: Phase1ValidationRuntimeConfig,
    per_code_profitability: Sequence[Phase1PerCodeProfitability] | None = None,
    thresholds: Phase1BehaviorQualityThresholds | None = None,
) -> Phase1LayerComputation:
    ...
```

`thresholds` 不传时使用 `Phase1BehaviorQualityThresholds()` 默认值；显式传入时用于
实验调参、测试和 checkpoint selector 复用同一套阈值配置。

## 5. Helper 设计

Helper 的设计目标是把 validation snapshot 中的原始数组转换成可解释的行为结构证据。
本层 helper 只负责数值计算、标签归类和基础输入处理，不负责 PASS/FAIL 判定，也不重新
执行交易收益。Layer 2 的盈利性字段只消费 Layer 3 产出的
`per_code_profitability`。

```python
def classify_market_morphology(
    prices: np.ndarray | None,
    *,
    fee_rate: float = 0.0002,
) -> np.ndarray:
    ...

def classify_action_motif(
    actions: np.ndarray,
    prices: np.ndarray | None = None,
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
    actions: np.ndarray,
    code_ids: np.ndarray,
    active_codes: Sequence[int],
    threshold: float,
) -> int:
    ...
```

实际实现可以把部分 helper 收敛为私有函数，只要入口
`compute_behavior_quality_metrics()` 的输出语义保持一致。

各 helper 的功能定义如下：

### 5.1 `classify_market_morphology`

将每条 validation horizon 的价格路径分类为市场形态标签。输入价格允许为
`[N, H]` 或 `[N, H, 1]`；内部应统一成二维 `[sample, horizon]`。如果价格缺失、
维度非法或 horizon 不足，应返回空标签数组，由主流程统一把 morphology 标为
`missing`，使 morphology 相关 gate 失败。

该 helper 应使用 validation set 自适应阈值，而不是写死绝对收益率阈值。至少应计算：

- 总收益 `total_return`；
- 前半段和后半段收益 `ret_first`、`ret_second`；
- realized volatility；
- range ratio；
- trend efficiency。

`fee_rate` 用于设置 neutral band 的下限，避免把小于交易成本噪声的价格波动误判为
趋势或反转。输出标签必须来自：

- `uptrend`
- `downtrend`
- `reversal-up`
- `reversal-down`
- `range-high-vol`
- `range-low-vol`
- `volatile-mixed`
- `neutral`

该 helper 的输出用于统计 `P(morphology | code)`、dominant morphology、
morphology purity 和 morphology lift。

### 5.2 `classify_action_motif`

将 decoded action sequence 映射为粗粒度交易 motif。动作 id 约定为：

- `0 = short`
- `1 = flat`
- `2 = long`

helper 内部先把动作映射为持仓值 `-1/0/1`，再从以下维度组合 motif：

- 主方向：`long`、`short`、`mixed`、`flat`；
- 入场时点：`early`、`middle`、`late`、`none`；
- 持仓风格：`hold`、`delayed-hold`、`brief-trade`、`switching`、`mostly-flat`；
- 可选价格关系或方向切换：`with-recent-move`、`against-recent-move`、
  `long-to-short`、`short-to-long`。

示例输出：

- `long + early + hold`
- `long + middle + delayed-hold + against-recent-move`
- `short + early + hold + long-to-short`
- `flat + none + mostly-flat`

motif 的目标不是还原每个 timestep，而是把同类交易意图归并，使同一 code 内的行为
一致性可以被统计。

### 5.3 `compute_distribution_by_code`

计算离散标签在每个 code 内的经验分布：

```text
values + code_ids -> code_id -> label -> probability
```

典型用途包括：

- `P(morphology | code)`；
- `P(motif | code)`；
- `P(morphology, motif | code)`。

实现时可以不保留完整分布，而是直接提取 dominant label、dominant ratio 和 entropy
purity。但语义上必须等价于先得到 per-code distribution，再由 distribution 计算
dominant 与 purity。

### 5.4 `compute_lift`

计算某个 code 内标签分布相对全局分布的富集程度：

```text
lift(label, code) = P(label | code) / P(label)
```

第一版只要求计算 dominant morphology 的 lift。分母必须加 `eps` 保护，避免全局分布
为 0 时产生除零。该指标用于区分“某 code 真的专注某类市场结构”和“该市场结构在全局
本来就占比很高”。

例如，全局 `uptrend` 占 20%，某 code 内 `uptrend` 占 50%，则该 code 对
`uptrend` 的 lift 为 2.5。

### 5.5 `compute_intra_code_action_similarity`

衡量同一 active code 内 decoded action sequence 是否一致。推荐实现使用 action
prototype，而不是对同一 code 内所有样本做两两比较：

1. 将 actions 映射为 positions；
2. 对每个 active code 计算平均持仓路径 prototype；
3. 计算样本与本 code prototype 的逐 timestep 差异；
4. 用 `1 - mean(abs(sample - prototype)) / 2` 转成相似度；
5. 对所有 active-code 样本求平均。

返回值越接近 1，说明 code 内行为越一致。没有有效 active code 时返回 `nan`。

### 5.6 `compute_inter_intra_separation`

衡量不同 active code 的行为中心是否明显分离。推荐口径为：

```text
mean_inter_distance / mean_intra_distance
```

其中：

- `mean_inter_distance` 是不同 code prototype 之间的平均距离；
- `mean_intra_distance` 是样本到自身 code prototype 的平均距离。

active code 少于 2 时返回 `nan`，并由 rule 层判定失败。该值越大，说明不同 archetype
在 decoded action 行为上越可区分。

### 5.7 `compute_duplicate_code_pair_count`

统计行为原型过度相似的 code pair 数量。每个 active code 先计算 action prototype，
再枚举无序 code pair。两个 prototype 的相似度可定义为：

```text
1 - mean(abs(left - right)) / 2
```

当相似度超过 `duplicate_code_similarity_max` 时，计为一个 duplicate pair。每对 code
只能计数一次。

### 5.8 其他内部 helper

实现中还可以包含以下私有 helper：

- `positions` helper：把 action id 映射为 `-1/0/1` 持仓值；
- `prices_2d` helper：统一价格数组形状；
- `active_codes` helper：根据 `runtime_config.active_code_min_occupancy` 过滤 active
  code；
- `entropy_purity` helper：计算 `1 - normalized entropy`，当 dominant ratio 不高但
  分布仍足够集中时提供补充证据；
- `approx_silhouette` helper：用 `z_e` 和 assigned code centroid 计算近似 latent
  silhouette；
- `per_code_profitability_map` helper：把 Layer 3 的 per-code profitability 转成
  `code_id -> profitability` 映射。

## 6. 市场形态分类

对每个 validation horizon 的价格序列 `p_0, ..., p_{h-1}`，先计算：

- `ret_total = p_{h-1} / p_0 - 1`
- `ret_first = p_mid / p_0 - 1`
- `ret_second = p_{h-1} / p_mid - 1`
- `realized_vol = std(log(p_t / p_{t-1})) * sqrt(h)`
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

- 主方向：`long`、`short`、`flat`、`mixed`；
- 入场时点：`early`、`middle`、`late`、`none`；
- 持仓风格：`hold`、`delayed-hold`、`brief-trade`、`switching`、`mostly-flat`；
- 方向切换：`long-to-short`、`short-to-long`；
- 与近期价格方向关系：`with-recent-move`、`against-recent-move`。

示例 motif：

- `long + early + hold`
- `long + middle + delayed-hold + against-recent-move`
- `short + early + hold + long-to-short`
- `flat + none + mostly-flat`

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
    weak_support_code_ratio_max: float = 0.20
    weak_structure_code_ratio_max: float = 0.40
    dominant_morphology_ratio_min: float = 0.35
    dominant_motif_ratio_min: float = 0.40
    motif_purity_min: float = 0.35
    dominant_pair_ratio_min: float = 0.30
    morphology_lift_min: float = 1.25
    morphology_purity_min: float = 0.30
    intra_code_similarity_min: float = 0.65
    inter_intra_separation_min: float = 1.30
    duplicate_code_similarity_max: float = 0.85
    latent_silhouette_score_min: float = 0.10
    profitable_code_coverage_min: float = 0.60
    duplicate_code_pair_count_max: int | None = None
```

Hard gates：

- `weak_support_code_ratio <= 0.20`
- `weak_morphology_code_ratio <= 0.40`
- `weak_motif_code_ratio <= 0.40`
- `weak_pair_code_ratio <= 0.40`
- `weak_lift_nonprofitable_code_ratio <= 0.40`
- `intra_code_action_similarity >= 0.65`
- `inter_intra_separation >= 1.30`
- `latent_silhouette_score >= 0.10`
- `duplicate_code_pair_count <= duplicate_code_pair_count_max`
- `profitable_code_coverage >= 0.60`

`duplicate_code_pair_count_max` 为 `None` 时，按当前 codebook size `K` 动态设置上限。
active code 少于 2 时，`inter_intra_separation` 和 `latent_silhouette_score` 写入
`nan`，应由 rule 层判定失败，因为 code 间分离度不可定义。

## 10. 缺失数据策略

- 缺少 `prices` 时，morphology 和 against/with recent move motif 不可靠；
  morphology 相关 hard gate 应 fail；
- 如果 Layer 3 尚未完成，`profitable_code_coverage` 写入 `nan`，正式 selector
  不应使用该 checkpoint；`weak_lift_nonprofitable_code_ratio` 可按“缺少盈利证据”
  保守统计；
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
