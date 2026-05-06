# Phase I Archetype Discovery 训练技术设计

## 1. 目标与范围

本文档定义 ArchetypeTrader 第一阶段 Archetype Discovery 的训练技术设计。Phase I 负责从离线数据处理产物中读取 demonstration trajectories，训练 VQ encoder-decoder 学习离散、可复用的 trading archetypes，并导出 codebook、decoder 和 horizon-level archetype labels 供 Phase II/III 使用。

Phase I 训练只做三件事：

1. 消费 `data_process_manifest.json` 加载离线数据处理阶段固化的 sampled horizons 与 DP teacher 产物。
2. 训练 VQ encoder-decoder，学习离散 archetype codebook 和因果 decoder。
3. 用 best checkpoint 导出 Phase II/III 所需产物：`encoder.pt`、`decoder.pt`、`codebook.pt`、`sampled_horizon_labels_*.feather`。

Phase I 训练不重新读取原始行情、不重新采样、不重新运行 DP。DP 只允许在离线数据处理阶段使用，Phase I/II/III 的验证、测试、回测推理和线上推理不能动态调用 DP。

## 2. 论文语义映射

| 论文概念 | 工程实现 |
| --- | --- |
| 采样 `n=30000` 个固定长度 chunk | 由离线数据处理阶段完成，训练通过 manifest 消费 |
| 每个 chunk 限制单次交易 | `SingleTradeDPPlanner` 在数据处理阶段已生成；工程允许 no-trade 样本作为低机会覆盖 |
| demonstration tuple `(s_demo, a_demo, r_demo)` | `sampled_horizons_{split}.feather.states` + `sampled_dp_teacher_{split}.feather.actions/rewards` |
| LSTM encoder 输出连续 latent | `ArchetypeEncoder`：单向 LSTM + MLP → `z_e` |
| VQ codebook 离散化 latent | `VectorQuantizer`：最近邻量化 + STE → `code_id`, `z_q` |
| decoder 根据 state 和 code 重构 action | `ArchetypeDecoder`：因果 LSTM，每步拼接 `z_q`，输出 3 分类 logits |
| 损失为 reconstruction + VQ + commitment | `Phase1Loss`：`L_rec + L_codebook + β₀·L_commit + λ_usage·L_usage + λ_tc·L_tc + λ_align·L_align` |
| Phase II 需要 demonstration label | best checkpoint 导出 `sampled_horizon_labels_*.feather.code_label` |

### 2.1 与论文第一阶段的一致性边界

设计保留论文核心架构：

```text
sampled horizons + DP demonstrations → LSTM encoder → VQ codebook → causal decoder reconstructs actions
```

以下属于工程落地增强，不改变论文主干：

| 设计项 | 与论文关系 | 是否改变核心公式 |
| --- | --- | --- |
| 数据处理与训练分离 | 论文未指定工程流程 | 否 |
| 手续费和盘口深度滑点 | 论文 MDP 已含 execution loss | 否 |
| `paper_formula` / `next_row_execution` reward 对齐 | `paper_formula` 严格对齐论文；`next_row_execution` 需单独标注 | `paper_formula` 不改变 |
| 因果 decoder 约束 | 约束实现，不改变公式 | 否 |
| Phase A 预训练 + Phase B VQ 训练 | 工程稳定性增强 | 否 |
| K-means warmup + EMA codebook 更新 | 工程稳定性增强 | 否 |
| dead-code restart / usage regularization | codebook collapse 防护 | 开启时扩展 loss |
| 可选 robust reward normalization | 工程增强；当前代码默认 standard | 否 |
| 时序对比学习 / 合成 horizon 增强 | 预留/ablation；当前未完整接入主链路 | 开启后会扩展 loss 或训练分布 |
| usage-profit alignment | 鼓励高收益 code 获得更高使用率 | 开启时增加 `L_align` |
| composite score + guardrail 体系 | 工程增强 | 否 |

### 2.2 当前实现与论文的差异标注

以下差异必须在实验报告或论文复现实验说明中明确标注，不能默认视为论文原设定：

| 项目 | 论文设定 | 当前工程默认/行为 | 标注要求 |
| --- | --- | --- | --- |
| commitment 权重 `β₀` | `0.25` | `ModelConfig.beta0=0.5`；`--paper-strict-reproduction` 才覆盖为 `0.25` | 主实验若不用 strict，应标注为工程增强超参 |
| 手续费率 | 实验设定 `δ=0.02%` | `CostConfig.commission_rate=0.0005`，即 `0.05%` | 会改变 DP teacher、student replay 和收益指标，必须标注 |
| codebook 更新 | 论文公式 (4) 是 gradient codebook loss | 默认 `update_method="ema"`，`ema_decay=0.95` | EMA 是稳定性增强；strict 模式改为 `gradient` |
| codebook 初始化 | 论文未指定 K-means warmup | 默认 `kmeans_warmup` | 作为工程增强或 ablation 标注 |
| Phase A 预训练 | 论文未定义 | 默认 `pretrain_epochs=10`，先跳过 VQ 做重构预训练 | strict 模式设为 `0` |
| usage regularization | 论文无 `L_usage` | 默认 `usage_regularization_weight=0.01` | 作为扩展 loss 标注 |
| usage-profit alignment | 论文无 `L_align` | 默认 `usage_profit_alignment_weight=0.05` | 作为扩展 loss 标注；strict 模式关闭 |
| reward 输入归一化 | 论文未指定 | 当前代码默认 `train_reward_standard` + `clip=5.0` | 设计曾考虑 robust；以代码默认为准 |
| no-trade horizon | 论文文字强调 profitable trajectory / single trade | 数据处理默认保留 no-trade/低机会覆盖并控制比例 | 训练集不是“全都有一次交易”，报告需输出 no-trade ratio |
| 样本数 `n=30000` | Phase I 实验采样 30k DP trajectories | 数据处理默认 `num_demos=30000`，训练只消费 manifest，不强制样本数 | manifest 中实际 count 才是准绳 |
| reward alignment | 论文等价 `paper_formula` | 工程还支持 `next_row_execution` | `next_row_execution` 不可直接声称论文复现 |

## 3. 数据契约

### 3.1 训练输入

Phase I 训练的唯一数据入口是 `data_process_manifest.json`：

```bash
python scripts/train_phase1.py \
  --pair FU \
  --train-batch-id phase1_vq_001 \
  --data-process-manifest artifacts/FU/batch_001/phase1/data_process_manifest.json
```

训练阶段通过 `Phase1ProcessedStore` 加载：

```python
store = Phase1ProcessedStore(artifact_dir)
manifest = store.load_manifest(path)
schema = store.load_schema(manifest)
train_records = store.load_records(manifest, "train")
val_records = store.load_records(manifest, "val")
test_records = store.load_records(manifest, "test")
```

必须校验：
- manifest `phase == "phase1_data_process"`
- manifest 包含 train/val/test 三个 split
- `input_schema.json` hash 等于 `schema_hash`
- sampled horizons 与 DP teacher 的 `sample_id` 集合完全一致
- `pair/split/_schema_hash/_data_process_hash/_dp_teacher_hash` 一致
- `actions/rewards` 长度等于 horizon 长度

训练阶段禁止调用：

```text
MarketFileReader.read_split
SlidingWindowIndexer.enumerate
StratifiedWindowSampler.sample
HorizonBuilder.build
Phase1DemoGenerator.generate
SingleTradeDPPlanner.plan
```

### 3.2 HorizonRecord 结构

每个样本长度 `h=72`：

| 字段 | shape | dtype | 说明 |
| --- | --- | --- | --- |
| `states` | `[h, feature_dim]` | `float32` | `feature_columns` 切出的模型输入，不含 `close` |
| `prices` | `[h+1]` 或 `[h+2]` | `float32` | `close` 切出的价格序列 |
| `execution_books` | `[h, levels, 4]` | `float32` | 每步成交行的 bid/ask 盘口 |
| `actions` | `[h]` | `int64` | DP teacher action `{0=short, 1=flat, 2=long}` |
| `rewards` | `[h]` | `float32` | 扣除手续费和滑点后的逐步净收益 |
| `sample_id` | scalar | `str` | 稳定样本 ID |
| `start_index` / `end_index` | scalar | `int64` | horizon 起止行 |
| `pair` / `split` | scalar | `str` | 标的与 split |

### 3.3 可选标签来源

除 sampled train/val/test 外，manifest 还可能携带额外的标签来源：

| manifest 字段 | 当前使用方式 | 后续含义 |
| --- | --- | --- |
| `full_time_sampled_horizons_path` / `full_time_dp_teacher_path` | 仅 train split 可选加载，导出 `sampled_horizon_labels_full_time_train.feather` | 离线分析/审计使用；当前 Phase II 主训练不消费 |
| `non_overlap_horizons_path` / `non_overlap_dp_teacher_path` | train/val/test 可选加载，导出 `non_overlap_horizon_labels_{split}.feather` | Phase II val/evaluation 与回测审计使用，降低 horizon 重叠泄漏 |

这些额外来源仍必须满足同一 schema/hash/sample_id/actions/rewards 长度校验；训练损失本身只使用 sampled train records。

### 3.4 状态字段

- `close` 保留为价格列，不进入模型输入状态特征列。
- `timestamp`、`symbol`、`split`、`sample_id` 等元信息列不进入模型。
- `input_schema.json` 必须同时记录 `price_column="close"`、`feature_columns` 和 `excluded_columns`，并保证 `close not in feature_columns`。
- Phase I 训练不在状态特征上拟合 scaler；`StateNormalizer` 只做 train-only robust z-score + clip。

## 4. 目录与模块设计

```text
scripts/train_phase1.py

src/phase1/config.py

src/preprocess_data/processed_store.py    # 训练加载 manifest/store

src/phase1/data/dataset.py
src/phase1/data/state_normalizer.py

src/phase1/models/vq_archetype.py
src/phase1/models/encoder_inputs.py
src/phase1/models/vector_quantizer.py
src/phase1/models/vq_losses.py

src/phase1/training/trainer.py
src/phase1/training/checkpoint.py
src/phase1/training/selection_policy.py

src/trading/env.py
src/trading/cost_model.py
src/trading/reward_alignment.py

src/phase1/evaluation/evaluator.py
src/phase1/evaluation/replay.py
src/phase1/evaluation/metrics.py
src/phase1/evaluation/report.py
src/evaluation/metrics/action.py
src/evaluation/metrics/risk.py
src/evaluation/metrics/archetype.py
src/evaluation/metrics/behavior.py
src/evaluation/metrics/stability.py
src/phase1/evaluation/diagnostics/latent_visualization.py
src/phase1/evaluation/diagnostics/failure_case_report.py

src/utils/feather_io.py
```

Phase I 训练主实现统一放在 `src/phase1/`。旧路径（如 `src/config/phase1_config.py`、`src/trainers/phase1_trainer.py`、`src/models/vq_archetype.py`）已删除，不再保留 re-export 兼容层；新增代码和测试必须引用 `src.phase1.*`。

### 4.1 `scripts/train_phase1.py`

CLI 入口，负责解析参数、加载配置、启动训练。

```bash
python scripts/train_phase1.py \
  --pair FU \
  --train-batch-id phase1_vq_001 \
  --data-process-manifest artifacts/FU/batch_001/phase1/data_process_manifest.json \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.001 \
  --device cuda
```

入口职责：
- 解析 CLI 参数并构建 `Phase1Config`
- 初始化日志、随机种子、产物目录
- 创建 `Phase1Trainer` 并调用 `run()`
- 捕获 `Phase1FatalError` 返回非零退出码

脚本层不得包含训练逻辑、不直接采样、不直接更新网络、不直接保存 checkpoint。

### 4.2 `src/phase1/config.py`

集中管理 Phase I 训练配置，使用 frozen dataclass + 显式 `_NESTED_TYPE_MAP`。

关键配置组：

```python
@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    code_dim: int = 16
    num_codes: int = 10
    beta0: float = 0.5

@dataclass(frozen=True)
class EncoderInputConfig:
    state_adapter_dim: int = 96
    action_embedding_dim: int = 16
    reward_embedding_dim: int = 16
    fusion_dim: int = 128
    reward_normalization: str = "train_reward_standard"
    reward_clip_value: float = 5.0

@dataclass(frozen=True)
class CodebookConfig:
    init_method: str = "kmeans_warmup"
    update_method: str = "ema"
    ema_decay: float = 0.95

@dataclass(frozen=True)
class CodebookHealthConfig:
    dead_code_restart: bool = True
    usage_regularization_weight: float = 0.01
    usage_profit_alignment_weight: float = 0.05

@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    pretrain_epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 256
    gradient_clip_norm: float = 1.0

@dataclass(frozen=True)
class SelectionPolicyConfig:
    metric_weights: dict = field(default_factory=lambda: {
        "switch_point_recall": 0.30,
        "switch_direction_accuracy": 0.20,
        "val_weighted_reconstruction_accuracy": 0.20,
        "val_return_capture_ratio": 0.20,
        "val_sharpe_ratio": 0.10,
    })

@dataclass(frozen=True)
class RiskGuardrailConfig:
    max_drawdown: float = 0.2
    min_sharpe_ratio: float = 0.0

@dataclass(frozen=True)
class BehaviorGuardrailConfig:
    min_inter_code_action_diversity: float = 0.15
    min_decoder_sensitivity_to_code: float = 0.05
    min_epoch_code_stability: float = 0.8
```

关键函数：
- `apply_paper_strict_overrides()` — `paper_strict_reproduction=True` 时关闭 usage_reg / usage-profit alignment / dead_code_restart / kmeans_warmup → random_normal / ema → gradient / beta0→0.25 / pretrain_epochs→0，并强制 reward normalization 为 `train_reward_standard` + `clip=5.0`
- `config_hash()` / `training_config_hash()` — 稳定 SHA256 前 16 位

### 4.3 `src/phase1/data/dataset.py`

PyTorch Dataset 适配层：

```python
class Phase1DemoDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        # 返回 states / actions / rewards(经normalizer) / trajectory_return(原始) / sample_id / contrastive_pair_id
```

边界约束：
- 不调用 DP，不执行分层采样，不计算 schema。
- normalizer transform 在 `__getitem__` 中即时应用，不修改原始 `rec.rewards`。

### 4.4 `src/phase1/data/state_normalizer.py`

Robust feature normalizer：

归一化策略：
1. 大额字段（turnover / volume / size）先做 `signed_log1p`
2. 所有特征做 robust z-score（median / MAD，1.4826 系数）
3. 近常量特征回退到 standard deviation，再不行 scale=1
4. clip 到 `[-8, 8]`

核心方法：
- `fit_records()` / `fit_matrix()` — train 数据拟合
- `transform_array()` — numpy array 变换
- `transform_records()` — HorizonRecord 列表 in-place 变换

### 4.5 `src/phase1/models/vq_archetype.py`

VQ Encoder-Decoder 模型：

```python
class ArchetypeEncoder(nn.Module):
    # 单向 LSTM + MLP 投影到 code_dim
    # 输出 z_e ∈ R^{code_dim}

class ArchetypeDecoder(nn.Module):
    # 因果 LSTM（bidirectional=False）
    # 每步拼接 z_q，输出 3 类 action logits
    # 严格约束：第 τ 步 logits 只依赖 s_{0:τ} 与 z_q

class VQArchetypeModel(nn.Module):
    # 整体模型 = input_adapter + encoder + quantizer + decoder
    def forward(self, states, actions, rewards) -> ModelOutputs
    def forward_pretrain(self, states, actions, rewards) -> ModelOutputs  # Phase A
    def encode(self, states, actions, rewards) -> (code_id, z_e)  # @torch.no_grad
    def decode(self, states_seq, code_id) -> (base_actions, decode_logits)  # Phase II/III
```

`ModelOutputs`：action_logits / z_e / z_q / z_q_no_grad / code_id

### 4.6 `src/phase1/models/encoder_inputs.py`

三路输入适配 + Reward 归一化：

```python
class RewardNormalizer:
    # fit_train(): robust (median/MAD) 或 standard (mean/std)
    # transform(): (x - center) / scale 后 clip
    # kurtosis 低于阈值自动回退 standard

class EncoderInputAdapter(nn.Module):
    # state → Linear + LayerNorm + GELU
    # action → Embedding(3, dim) + LayerNorm
    # reward → Linear(1, dim) + LayerNorm + GELU
    # concat → Linear(sum → fusion_dim) + LayerNorm
```

### 4.7 `src/phase1/models/vector_quantizer.py`

VQ-VAE 风格 codebook：

```python
class VectorQuantizer(nn.Module):
    def warmup_initialize(self, z_e_samples)  # K-means++ / sample_encoder_outputs / random_normal
    def quantize(self, z_e) -> QuantizeOutput  # 最近邻 + STE
    def update_codebook(self, z_e, code_id)  # EMA 模式更新（含 Laplace smoothing）
    def restart_dead_codes(self, z_e_high_error)  # 从高重构误差样本重置
    def usage_stats(self) -> CodeUsageStats  # counts / usage_ratio / perplexity / dominant_ratio / dead_codes
```

### 4.8 `src/phase1/models/vq_losses.py`

```python
class Phase1Loss(nn.Module):
    # L = L_rec + L_codebook + β₀·L_commit + λ_usage·L_usage + λ_tc·L_tc + λ_align·L_align
    # reconstruction: cross_entropy(logits, targets)
    # codebook: ||sg[z_e] - z_q||²
    # commitment: ||z_e - sg[z_q]||²
    # usage: KL(U(K) || p_code)
    # contrastive: cosine similarity (同 pair_id)
    # alignment: usage-profit alignment

    def forward_pretrain(self, ...)  # Phase A 只计算 action reconstruction CE
```

### 4.9 `src/phase1/training/trainer.py`

主编排器，完整训练流程：

```python
class Phase1Trainer:
    def run(self) -> TrainerArtifacts
    def _build_training_components(self)
    def _train_loop(self)  # Phase A 预训练 + Phase B VQ 训练
    def _warmup_codebook(self)  # K-means 初始化
    def _maybe_restart_dead_codes(self)
    def _export_horizon_labels(self)  # best checkpoint → code labels
    def _export_phase2_artifacts(self)  # encoder.pt / decoder.pt / codebook.pt
    def _build_sampling_leakage_diagnostics(self)
    def _build_signoff_diagnostics(self)
```

训练分两阶段：
- **Phase A（pretrain）**：跳过 VQ，用 `z_e` 直接条件化 decoder，只做 action reconstruction。建立 encoder/decoder 基础表示。
- **Phase B（VQ 训练）**：完整 VQ 训练 + EMA codebook 更新 + dead code restart。

实现细节：
- `__init__` 中调用 `apply_paper_strict_overrides`
- AMP 混合精度有安全检查：输入量级超过 `1e4` 时自动回退 fp32
- 每 5 个 epoch 评估一次，Phase A 结束时额外跑一次 full validation 建基线
- 使用 CosineAnnealingLR 学习率调度器
- 捕获 `Phase1FatalCollapse` 并转抛 `Phase1FatalError`

### 4.10 `src/phase1/training/checkpoint.py`

Checkpoint 持久化：

```python
class Phase1CheckpointManager:
    def save_last(self, state, metrics)
    def save_periodic(self, state, metrics, epoch)
    def commit_verdict(self, verdict)  # 由 selection_policy 决定是否 promote 为 best
    def load(self, path)
```

- 所有文件写入使用原子操作（tmp + os.replace）
- 每条 entry 计算文件 sha256 用于审计
- 不嵌入 best 选择规则或 guardrail

### 4.11 `src/phase1/training/selection_policy.py`

Best checkpoint 选择策略：

```python
class Phase1SelectionPolicy:
    def evaluate(self, metrics, history) -> SelectionVerdict
    # 7 步决策：cooldown → codebook guardrail → risk guardrail → behavior guardrail
    # → teacher warning → composite score 比较 → consecutive collapse fatal
```

决策类型：`promote_to_best` / `reject` / `keep_as_periodic` / `fatal` / `skipped`

Guardrail 体系：
- **codebook guardrail**：`code_usage_ratio < 0.7` → reject
- **risk guardrail**：`max_drawdown > 0.2` 或 `sharpe < 0.0` → reject
- **behavior guardrail**：`inter_code_diversity < 0.15` / `decoder_sensitivity < 0.05` / `epoch_stability < 0.8` → reject
- **teacher quality guardrail**：仅 warning，不 reject

### 4.12 `src/phase1/evaluation/evaluator.py`

评估编排器，9 步评估：

1. 模型 forward → action 指标
2. VQ 指标
3. teacher/student replay
4. 风险指标
5. DP teacher quality
6. per-archetype 指标
7. 行为多样性
8. epoch code stability
9. warnings 汇总

### 4.13 `src/phase1/evaluation/replay.py`

三种 replay：
- `replay_dp_teacher()` — DP teacher actions → env replay
- `replay_student_online()` — decoder + codebook → 因果 streaming → env replay
- `evaluate_horizon_boundaries()` — 跨 horizon 仓位继承 → 边界换仓成本

### 4.14 `src/phase1/evaluation/metrics.py`

指标门面 + 组合指标：

```python
def phase1_composite_score(metrics, weights) -> (score, debug_info)
def composite_score_sensitivity(metrics_history, weights, perturbations)
def composite_score_sensitivity_across_epochs(metrics_history, weights)
```

### 4.15 `src/phase1/evaluation/report.py`

统一 report 写入，必填字段 30+ 个，落盘前校验 schema。

### 4.16 `src/trading/env.py`

统一分钟级交易环境：

```python
class TradingEnv:
    def reset(self, horizon, initial_position=0)
    def step(self, action)  # action {0,1,2} → {-max_position, 0, +max_position}
    def replay(self, actions)  # 一次性 replay 整段动作序列
```

### 4.17 `src/trading/cost_model.py`

盘口逐档成交成本模型：

```python
class LobDepthCostModel:
    def execute(self, prev_position, target_position, execution_book, mark_price) -> ExecutionResult
    # 逐档累加成交；深度不足 → reject_transition
    # fee = commission_rate * |Δ| * mark_price
    # slippage = |Δ| * |fill_price - mark_price|
```

### 4.18 `src/trading/reward_alignment.py`

```python
class RewardAlignment:
    # paper_formula: decision=t, execution=t, markout=t+1
    # next_row_execution: decision=t, execution=t+1, markout=t+2
```

## 5. VQ Encoder-Decoder 设计

### 5.1 Encoder

```text
state_t → state_adapter(Linear + LayerNorm + GELU)
action_t → action_embedding(3, dim) + LayerNorm
reward_t → reward_normalizer + reward_adapter(Linear + LayerNorm + GELU)
concat(state_emb, action_emb, reward_emb) → fusion_layer(Linear + LayerNorm)
→ LSTM(hidden_dim=128) → last hidden → MLP → z_e(16)
```

Reward normalization 当前代码默认 `train_reward_standard` + `clip=5.0`。如果实验切换到 `train_reward_robust`，使用：

$$\hat{r}_t = \text{clip}\left(\frac{r_t - \text{median}(r)}{1.4826 \cdot \max(\text{MAD}(r), \epsilon)}, -c, c\right)$$

当配置为 `train_reward_robust` 且 train reward kurtosis 低于 6.0 时自动回退 `train_reward_standard`，回退原因写入 report。

健康指标：
- `reward_norm_clip_ratio`：被 clip 的 reward 比例
- `encoder_input_modality_norms`：state/action/reward embedding 平均 L2 norm
- `reward_embedding_norm_ratio`：reward_emb_norm / state_emb_norm

### 5.2 Vector Quantizer

codebook $\epsilon = \{e_0, \ldots, e_{K-1}\}$，$K=10$。

最近邻量化 + STE：

```python
z_q_st = z_e + (z_q - z_e).detach()
```

初始化方式：

| `init_method` | 说明 | 适用 |
| --- | --- | --- |
| `random_normal` | 随机初始化 | 最简单，早期易 dead code |
| `sample_encoder_outputs` | 用首批 z_e 初始化 | 比随机稳定 |
| `kmeans_warmup` | K-means++ 初始化 | 默认推荐 |

更新方式：

| `update_method` | 说明 | 与论文关系 |
| --- | --- | --- |
| `gradient` | codebook loss 直接梯度更新 | 严格贴近论文公式 (4) |
| `ema` | EMA 滑动均值更新 | 工程稳定性更好 |

EMA 更新：

$$N_i \leftarrow \lambda N_i + (1-\lambda) n_i$$
$$m_i \leftarrow \lambda m_i + (1-\lambda) \sum_{b:k_b=i} z_{e,b}$$
$$e_i \leftarrow \frac{m_i}{N_i + \epsilon}$$

### 5.3 Decoder

```text
state projection → concat repeated z_q at each timestep
→ unidirectional LSTM(batch_first=True)
→ timestep-wise MLP head → action logits [batch, h, 3]
```

严格因果约束：
- LSTM 必须是 `bidirectional=False`
- 第 τ 步 logits 只依赖 `s_{0:τ}` 和 `z_q`
- 禁止使用整段 horizon pooling、未来收益或 bidirectional 结构
- 单元测试必须验证：修改 `s_{τ+1:h-1}` 不改变第 τ 步 logits

### 5.4 Loss

$$L = L_{rec} + \|sg[z_e] - z_q\|^2 + \beta_0 \|z_e - sg[z_q]\|^2 + \lambda_{usage} L_{usage} + \lambda_{tc} L_{tc} + \lambda_{align} L_{align}$$

- `L_rec`：action reconstruction cross entropy
- `L_codebook`：推动 codebook 向 z_e 移动
- `L_commit`：推动 z_e 向选中 code 收敛，默认 `β₀=0.5`
- `L_usage`：`KL(U(K) || p_code)`，鼓励均匀使用
- `L_tc`：cosine contrastive loss（同 pair_id 的 z_e 距离）
- `L_align`：usage-profit alignment，鼓励高收益 code 获得更高使用率

Phase A 只计算 `L_rec`。

### 5.5 Codebook Collapse 防护

| 指标 | 告警阈值 | 处理 |
| --- | --- | --- |
| `code_usage_ratio` | `< 0.7` | checkpoint 不可成为 best，启动 usage regularization 或 dead-code restart |
| `perplexity` | 长期接近 1.0 | 降低 lr、增加 commitment 调节 |
| `dead_code_count` | 连续多 epoch 使用次数为 0 | 从高误差样本重置 code |
| `dominant_code_ratio` | 单个 code 占比 `> 0.5` | 提高 usage regularization 权重 |

Dead-code restart 策略：
1. 统计每个 code 连续未被使用的 epoch 数
2. 超过 `dead_code_patience` 未被使用，从高重构误差样本抽取 encoder 输出
3. 用抽取到的 `z_e` 重置该 code，同步重置 EMA buffer
4. 记录到 `phase1_report.json` 和 `checkpoint_manifest.json`

当 dead-code restart 在 `consecutive_collapse_epoch_limit` 内连续触发但 `code_usage_ratio` 仍 `< 0.7` 时，trainer 必须以非零退出码停止训练并写入 `fatal_collapse=true`。

### 5.6 未来信息与因果性边界

| 组件 | 是否可看完整 72 行 | 进入线上决策 |
| --- | --- | --- |
| Single-trade DP planner | 可以（hindsight expert） | 否 |
| VQ encoder | 可以（离线 label 生成） | 否 |
| VQ decoder | 不可以看未来 | 是 |
| Phase II selector | 不可以看未来 | 是 |

Phase I 允许用未来信息构造"老师答案"，但不允许训练出在执行时依赖未来信息的学生模型。

## 6. 训练流程

```text
data_process_manifest.json
  → Phase1ProcessedStore 加载 train/val/test records
  → StateNormalizer.fit_records(train) → transform all splits
  → RewardNormalizer.fit_train(train rewards) → transform all splits
  → Phase1DemoDataset + DataLoader
  → VQArchetypeModel + Phase1Loss + AdamW + CosineAnnealingLR
  → Phase A: pretrain (跳过 VQ, 只做 action reconstruction)
  → K-means warmup codebook
  → Phase B: VQ 训练 + EMA codebook + dead code restart
      每 5 epoch: Phase1Evaluator.evaluate_epoch()
      → Phase1SelectionPolicy.evaluate() → SelectionVerdict
      → Phase1CheckpointManager.commit_verdict()
  → best checkpoint 冻结
  → _export_horizon_labels() → sampled_horizon_labels_{split}.feather
  → _export_phase2_artifacts() → encoder.pt / decoder.pt / codebook.pt
  → composite_score_sensitivity_across_epochs()
  → Phase1ReportWriter.write_final_report()
  → _build_signoff_diagnostics()
```

详细步骤：

1. 校验 manifest 完整性与 hash 一致性。
2. 加载 train/val/test records，校验 `sample_id` 集合一致。
3. `StateNormalizer.fit_records(train)` → `state_normalizer.json` → transform all splits。
4. `RewardNormalizer.fit_train(train_rewards)` → `reward_normalizer.json` → transform all splits。
5. 构建 `Phase1DemoDataset` + `collate_phase1`。
6. 实例化 `VQArchetypeModel` + `Phase1Loss` + optimizer + scheduler。
7. **Phase A**：pretrain `pretrain_epochs` 个 epoch，跳过 VQ，用 `z_e` 直接条件化 decoder。
8. Phase A 结束后跑 full validation 建基线。
9. **K-means warmup**：用 train batches 的 `z_e` 初始化 codebook。
10. **Phase B**：完整 VQ 训练，每 epoch 检查 dead code 并可能 restart。
11. 每 5 个 epoch 调用 `Phase1Evaluator.evaluate_epoch()`。
12. `Phase1SelectionPolicy.evaluate()` 决定 promote/reject/fatal。
13. `Phase1CheckpointManager.commit_verdict()` 执行 checkpoint IO。
14. 训练结束后，用 best checkpoint 导出 horizon labels 和 Phase II 产物。
15. 跑 `composite_score_sensitivity_across_epochs()`。
16. `Phase1ReportWriter.write_final_report()` 写入最终报告。

## 7. 评估指标体系

### 7.1 Action 指标

| 指标 | 说明 |
| --- | --- |
| `reconstruction_accuracy` | action 重构准确率（sanity check） |
| `weighted_reconstruction_accuracy` | 按类别权重加权 |
| `non_flat_accuracy` | short/long 非 flat timestep 重构准确率 |
| `cross_entropy` | action 重构 CE loss |
| `switch_point_recall` | 切换点召回率 |
| `switch_direction_accuracy` | 切换方向准确率 |
| `single_trade_consistency_rate` | decoder 输出满足单次切换约束的比例 |

### 7.2 VQ 指标

| 指标 | 说明 |
| --- | --- |
| `code_usage_ratio` | 被使用 code 数 / K |
| `perplexity` | codebook perplexity |
| `inter_code_distance` | codebook 向量间平均距离 |
| `silhouette_score` | latent space 按 code_id 分组的轮廓系数 |
| `epoch_code_stability` | best/last epoch code 分配一致率 |

### 7.3 Replay 收益指标

| 指标 | 说明 |
| --- | --- |
| `val_student_online_net_return` | student 因果 replay 净收益 |
| `val_dp_teacher_net_return` | DP teacher 净收益 |
| `val_return_capture_ratio` | student / max(abs(teacher), eps) |
| `val_regret_to_dp` | teacher - student |
| `val_cost_paid` | 手续费 + 滑点总和 |

### 7.4 风险调整收益指标

| 指标 | 说明 |
| --- | --- |
| `val_sharpe_ratio` | 年化 Sharpe (annualization_factor=525600) |
| `val_sortino_ratio` | 年化 Sortino |
| `val_max_drawdown` | equity curve 最大回撤 |
| `val_calmar_ratio` | 年化收益 / 最大回撤 |

### 7.5 行为多样性指标

| 指标 | 说明 |
| --- | --- |
| `per_code_action_entropy` | 每个 code 解码出的 action 分布熵 |
| `inter_code_action_diversity` | 固定 states 后不同 code 输出 action 序列的平均 Hamming 距离 |
| `decoder_sensitivity_to_code` | 固定 states，替换 code_id 后 decoder logits 变化幅度 |

### 7.6 边界衔接指标

| 指标 | 说明 |
| --- | --- |
| `horizon_boundary_turnover_cost` | 相邻 horizon 边界换仓成本 |
| `horizon_boundary_position_consistency` | 边界仓位一致比例 |

### 7.7 DP Teacher 质量指标

| 指标 | 说明 |
| --- | --- |
| `val_dp_teacher_sharpe` | DP teacher Sharpe |
| `val_dp_teacher_profitable_ratio` | DP teacher 正收益 horizon 占比 |

### 7.8 Composite Score

$$\text{composite} = 0.30 \cdot \text{switch\_recall} + 0.20 \cdot \text{switch\_dir\_acc} + 0.20 \cdot \text{weighted\_recon} + 0.20 \cdot \text{capture\_ratio} + 0.10 \cdot \text{sharpe}$$

主实验完成后必须做权重 sensitivity 检查，写入 `composite_score_sensitivity.json`。

## 8. Checkpoint 选择与 Guardrail

### 8.1 选择流程

```text
epoch metrics → Phase1SelectionPolicy.evaluate()
  → cooldown check (dead-code restart 后 N epoch 冷却)
  → codebook guardrail (code_usage_ratio < 0.7 → reject)
  → risk guardrail (max_drawdown / sharpe → reject)
  → behavior guardrail (diversity / sensitivity / stability → reject)
  → teacher quality warning (仅 warning)
  → composite score 比较
  → consecutive collapse fatal
→ SelectionVerdict
→ Phase1CheckpointManager.commit_verdict()
```

### 8.2 Guardrail 体系

| Guardrail | 阈值 | 行为 |
| --- | --- | --- |
| code_usage_ratio | `< 0.7` | reject |
| max_drawdown | `> 0.2` | reject |
| min_sharpe_ratio | `< 0.0` | reject |
| inter_code_action_diversity | `< 0.15` | reject |
| decoder_sensitivity_to_code | `< 0.05` | reject |
| epoch_code_stability | `< 0.8` | reject |
| dp_teacher_profitable_ratio | `< 0.3` | warning only |

### 8.3 Fatal Collapse

当 `dead_code_restart` 在 `consecutive_collapse_epoch_limit` 内连续触发但 `code_usage_ratio` 仍不达标时：
- trainer 以非零退出码停止训练
- `phase1_report.json` 写入 `fatal_collapse=true`、`code_assignment_drift_warning=true`
- `best_vq_model.pt` 不可被声明为 sign-off 版本

## 9. 具体产出物契约

```text
artifacts/{PAIR}/{TRAIN_BATCH_ID}/phase1/
```

Phase I 训练完成后产出分为四类：

1. **Phase II/III 必需接口产物**：冻结 decoder、codebook、schema、horizon-level labels。
2. **Phase I 复现产物**：配置、normalizer、完整 checkpoint、checkpoint manifest。
3. **审计/签收产物**：report、composite sensitivity、sampling leakage/sign-off diagnostics。
4. **可选标签来源产物**：full-time train labels、non-overlap labels。

### 9.1 Phase II/III 最小消费集合

Phase II selector 启动训练至少需要：

| 文件 | 是否必需 | 消费方 | 说明 |
| --- | --- | --- | --- |
| `decoder.pt` | 必需 | Phase II/III frozen policy | 从 best checkpoint 导出的冻结 causal decoder；输入 state 序列 + selected code，输出 base action logits |
| `codebook.pt` | 必需 | Phase II selector action space | `K x code_dim` archetype embedding；Phase II action id 必须落在 `[0, K-1]` |
| `input_schema.json` | 必需 | Phase II dataset / state builder | 固化 `feature_columns`、`price_column`、`excluded_columns`；Phase II 必须按同一特征顺序构造 state |
| `state_normalizer.json` | 必需 | Phase II dataset / replay | Phase I train-only state normalizer；Phase II 复用同一变换，避免特征尺度漂移 |
| `sampled_horizon_labels_train.feather` | 必需 | Phase II KL/demo regularization | sampled train horizon 的 `code_label`；Phase II 训练只读取该 train label |
| `non_overlap_horizon_labels_val.feather` | 必需 | Phase II validation/evaluation | non-overlap val horizon 的 `code_label`；Phase II 评估只读取该 val label |
| `phase1_config.yaml` | 必需 | Phase II artifact validator | 读取 `horizon`、`max_position`、`reward_alignment`、成本参数，校验 Phase I/II 执行语义一致 |
| `phase1_report.json` | 必需 | Phase II sign-off / dead-code mask | 读取 Phase I sign-off、code usage、hindsight warning 等审计信息 |

Phase II 的当前工程契约固定为：train split 使用 `sampled_horizons_train.feather` + `sampled_horizon_labels_train.feather` 训练；val/evaluation 使用 `non_overlap_horizons_val.feather` + `non_overlap_horizon_labels_val.feather` 评估。Phase II 主训练不加载 test split market data，也不加载任何 test label；`phase1_label_source` 仅为旧配置兼容字段，不再改变 train label 来源。

### 9.2 全量文件清单

| 文件 | 类别 | 作用 | 后续使用方 |
| --- | --- | --- |
| `phase1_config.yaml` | 复现/接口 | 固化训练配置和成本语义 | 复现、Phase II/III 上下文 |
| `input_schema.json` | 接口 | 字段契约和 feature 顺序 | Dataset、Phase II/III 对齐 |
| `feature_provenance.json` | 审计 | 记录特征来源与数据预处理阶段 provenance | 复现、数据审计 |
| `state_normalizer.json` | 接口/复现 | 状态特征归一化参数 | Phase I 复现、Phase II state transform |
| `reward_normalizer.json` | 复现 | encoder reward 归一化参数 | Phase I label 复现、离线分析 |
| `encoder.pt` | 模型 | encoder 权重 | 离线编码、分析；线上不使用 |
| `decoder.pt` | 接口/模型 | 冻结 decoder 权重 | Phase II/III 推理 base action |
| `codebook.pt` | 接口/模型 | K=10 code embedding | Phase II selector 动作空间 |
| `best_vq_model.pt` | 复现/checkpoint | 完整最优 checkpoint | 继续训练、回滚、重新导出 |
| `last_vq_model.pt` | checkpoint | 最后 epoch checkpoint | 断点恢复、失败复盘 |
| `checkpoints/epoch_*.pt` | checkpoint | 周期性 checkpoint | 调试、ablation |
| `checkpoint_manifest.json` | 审计/checkpoint | checkpoint sha、metrics、selection verdict、best 选择记录 | 审计、复现 |
| `sampled_demos_train.feather` | 复现/审计 | 训练 demonstrations 快照，含 states/actions/rewards/meta | 训练复盘、demo 审计 |
| `sampled_horizon_labels_train.feather` | 接口/label | sampled train code labels | Phase II KL/demo regularization |
| `sampled_horizon_labels_val.feather` | 分析/label | sampled val code labels | 离线分析；Phase II 主评估使用 non-overlap val labels |
| `sampled_horizon_labels_test.feather` | 分析/label | sampled test code labels | 离线分析；Phase II 主训练不得用 test label |
| `sampled_horizon_labels_full_time_train.feather` | 可选 label | 可选 full-time train code labels | 离线分析/审计；当前 Phase II 主训练不消费 |
| `non_overlap_horizon_labels_{split}.feather` | 可选/接口 label | 可选非重叠 code labels | `val` 为 Phase II 评估必需；`train/test` 用于回测审计与泄漏控制 |
| `sampled_horizons_full_time_train.feather` / `sampled_dp_teacher_full_time_train.feather` | 可选审计拷贝 | full-time train 标签来源拷贝 | label provenance 审计 |
| `phase1_report.json` | 审计/sign-off | 训练与诊断汇总、best epoch、guardrail、Phase II eligible 状态 | 验收、实验对比、Phase II sign-off |
| `composite_score_sensitivity.json` | 审计 | 权重敏感性检查 | checkpoint 选择审计 |
| `sampling_leakage_diagnostics.json` | 审计/sign-off | 后视/前瞻对照与 sign-off 阻塞原因 | sign-off |
| `action_diagnostics.json` | 可选诊断 | best epoch action 混淆矩阵、precision/recall、switch timing | 误差分析 |
| `risk_diagnostics.json` | 可选诊断 | best epoch Sharpe、Sortino、MDD、Calmar 等风险诊断 | 风险复盘 |
| `archetype_separation.json` | 可选诊断 | code usage、perplexity、silhouette、per-code metrics | codebook 健康分析 |
| `archetype_behavior_diagnostics.json` | 可选诊断 | inter-code diversity、decoder sensitivity、no-trade concentration | archetype 行为分析 |
| `horizon_boundary_diagnostics.json` | 可选诊断 | horizon 边界换仓成本与仓位一致性 | Phase II 边界风险评估 |
| `code_stability_diagnostics.json` | 可选诊断 | epoch code stability、codebook displacement | label 稳定性复盘 |
| `tensorboard/` | 规划中 | 当前 trainer 未接入 `SummaryWriter` | 观察 codebook 演化 |
| `latent_snapshots/` | 规划中 | 模块存在但当前 trainer 未调用 | 离线复盘 |
| `failure_cases/` | 规划中 | 模块存在但当前 trainer 未调用 | 定位问题 |

### 9.3 Label 文件字段

`sampled_horizon_labels_*.feather`、`sampled_horizon_labels_full_time_train.feather` 与 `non_overlap_horizon_labels_{split}.feather` 字段一致：

| 字段 | 说明 |
| --- | --- |
| `sample_id` | horizon ID |
| `start_index` / `end_index` | 起止位置 |
| `last_execution_row` / `last_markout_row` | reward alignment 审计 |
| `strata_label` | 分层标签 |
| `code_label` | VQ encoder 分配的 archetype ID |
| `demo_return` | DP demonstration horizon return |
| `num_switches` | action 切换次数 |
| `is_no_trade` | 是否全程 flat |
| `sample_source` | label 来源，如 sampled / full_time / non_overlap |
| `_config_hash` | Phase I config hash，用于防止错配旧 label |
| `_schema_hash` | input schema hash，用于防止特征字段错配 |

### 9.4 产物生成顺序和失败边界

产物生成顺序：

```text
phase1_config.yaml / input_schema.json / feature_provenance.json
  → state_normalizer.json / reward_normalizer.json
  → sampled_demos_train.feather
  → last_vq_model.pt / checkpoints/epoch_*.pt / checkpoint_manifest.json
  → best_vq_model.pt
  → sampled_horizon_labels_{train,val,test}.feather
  → sampled_horizon_labels_full_time_train.feather (可选)
  → non_overlap_horizon_labels_{train,val,test}.feather (可选)
  → encoder.pt / decoder.pt / codebook.pt
  → phase1_report.json / diagnostics json
```

失败边界：
- 如果训练没有产生 `best_vq_model.pt`，禁止导出 `encoder.pt`、`decoder.pt`、`codebook.pt` 和 label 文件。
- 如果 sign-off 因 prospective diagnostic 缺失或 hindsight bias 超阈值被阻塞，模型产物可以存在，但 `phase1_report.json.phase1_checkpoint_eligible_for_phase2` 必须为 false，Phase II 主实验不得默认消费。
- `encoder.pt` 只服务离线编码/复盘；Phase II/III 线上路径只使用 `decoder.pt` + `codebook.pt`。

## 10. 验收标准

### 10.1 数据验收

- 必须通过 `data_process_manifest.json` 加载数据，不重新读取原始行情。
- `sample_id` 集合在 sampled horizons 与 DP teacher 间完全一致。
- `close` 不出现在 `feature_columns` 或 `states` 中。
- `StateNormalizer` 只在 train 上 fit，val/test 只 transform。
- `RewardNormalizer` 只在 train rewards 上 fit，val/test 只 transform。

### 10.2 VQ 验收

- `phase1_composite_score` 达到配置阈值。
- `code_usage_ratio >= 0.7`。
- `perplexity` 不能长期塌缩到 1。
- dead-code restart 默认开启，记录 `dead_code_restarts`。
- 连续 collapse 时训练以非零退出码结束。
- encoder 使用三路 adapter + reward normalization，禁止 raw concat。
- decoder 因果性：修改 `s_{τ+1:h-1}` 不改变第 τ 步 logits。
- teacher 和 student replay 使用同一 `TradingEnv` + `CostModel`。

### 10.3 Checkpoint 验收

- `best_vq_model.pt` 对应 `checkpoint_manifest.json` 中 `is_best=true` 的 epoch。
- `code_usage_ratio < 0.7` 的 checkpoint 不能成为 best。
- `val_max_drawdown > 0.2` 的 checkpoint 不能成为 best。
- `encoder.pt`、`decoder.pt`、`codebook.pt` 必须从 `best_vq_model.pt` 导出。
- 必须完成 `composite_score_sensitivity` 检查。

### 10.4 产物验收

- Phase II 可以只依赖 `decoder.pt`、`codebook.pt`、`sampled_horizon_labels_train.feather`、`non_overlap_horizon_labels_val.feather`、`input_schema.json`、`state_normalizer.json`、`phase1_config.yaml` 启动主训练；test label 不进入训练。
- 固定 seed + 固定 manifest 后，重复运行得到一致的 `best_checkpoint_path` 和可比指标。

## 11. 风险与处理

| 风险 | 表现 | 处理 |
| --- | --- | --- |
| codebook collapse | 大部分样本落到同一个 code | dead_code_restart + usage_regularization + consecutive_collapse_epoch_limit 强制中止 |
| decoder 忽略 code | 不同 code 解码动作几乎相同 | `inter_code_action_diversity` / `decoder_sensitivity_to_code` guardrail |
| reward 重尾被 standard normalization 削弱 | 大行情切换点信号被截断 | 当前代码默认 `train_reward_standard` + `clip=5.0`；如需抗重尾应显式切换到 `train_reward_robust` 并做 ablation |
| composite score 固定权重过拟合 selection | 不同权重下 best epoch 漂移 | 权重 sensitivity 检查 + `composite_weight_sensitivity_warning` |
| markout 行越过 split 边界 | `last_markout_row` 超出文件行数 | 数据处理阶段已裁掉；训练阶段校验 |
| K-means + EMA 卡在局部最优 | codebook 过早锁死 | 独立 BATCH_ID 做 seed/初始化对照；必要时低概率扰动 |
| 后视分层导致验证虚高 | horizon 内统计量采样 | 数据处理阶段已做 prospective 对照；训练阶段读取 `sampling_leakage_diagnostics.json` |
| encoder reward 信号被淹没 | reward_t 量级远小于状态特征 | reward_normalizer + reward adapter；监控 `reward_embedding_norm_ratio` |
| 快速验证误导 checkpoint | fast probe 指标好但 full validation 差 | best candidate 必须跑 full validation；报告区分 fast/full |
| AMP 数值不稳定 | 混合精度下 loss 异常 | 输入量级超过 1e4 自动回退 fp32 |

## 12. 与 Phase II 的接口

Phase II 读取：
- `decoder.pt`、`codebook.pt`
- train: `sampled_horizons_train.feather` + `sampled_horizon_labels_train.feather`
- val/evaluation: `non_overlap_horizons_val.feather` + `non_overlap_horizon_labels_val.feather`
- `input_schema.json`

Phase II 使用 `code_label` 作为 KL/demo regularization 的 ground-truth：

$$\hat{a}^{sel}_t = code\_label_t$$

Phase II 推理流程：

```text
selector chooses code_id
  → frozen causal decoder receives current/past states and codebook[code_id]
  → decoder emits the current step base action
```

Phase II 必须处理 horizon 间仓位连续性：
- `HorizonEnv.reset()` 接收 `prev_terminal_position`
- 第一步 target position 与 inherited position 不一致时，通过同一 `CostModel` 扣除换仓成本
- Phase II 评估报告继续输出 `horizon_boundary_turnover_cost` 和 `horizon_boundary_position_consistency`

Phase I 的最终验收不是单纯 reconstruction accuracy，而是能否提供稳定、可复用、可解释的 discrete archetype interface。

## 13. 本轮审核补充与待明确事项

### 13.1 已确认需要写入设计的工程事实

| 事项 | 当前代码事实 | 设计约束 |
| --- | --- | --- |
| manifest-only 训练 | `Phase1Trainer.run()` 只通过 `Phase1ProcessedStore` 读取已固化 horizons/teacher | 训练阶段禁止重新采样和在线 DP |
| 额外标签来源 | trainer 会导出可选 `full_time_train` 与 `non_overlap_*` labels | Phase II 主训练固定使用 sampled train；主评估固定使用 non-overlap val；full-time 仅作离线审计 |
| sign-off 阻塞 | 训练可完成但因 prospective diagnostic 缺失或 hindsight delta 超阈值抛 `Phase1FatalError` | “训练完成”不等于“可作为 Phase II 主实验输入” |
| reward 保真 | dataset 即时 transform rewards，`rec.rewards` 保留原始值 | `demo_return`、teacher replay 和收益报告必须使用原始 reward |
| best 选择 | Phase A baseline validation 只建稳定性基线，不参与 best | best checkpoint 必须来自 Phase B VQ 训练 |
| AMP 安全 | CUDA mixed precision 会按 batch 输入量级自动回退 fp32 | 报告/日志中保留 AMP 回退 warning，避免把 NaN 归因于模型结构 |

### 13.2 当前“配置/模块存在但未完整接入”的项

这些项不应在设计文档中表述为已经完整生产可用；开启前需要补实现或做单独验证：

| 项目 | 代码状态 | 建议 |
| --- | --- | --- |
| temporal contrastive augmentation / `L_tc` | preprocessing 中构造 pair 的代码当前注释掉；trainer 创建 `Phase1DemoDataset(..., contrastive_pairs=None)`，即使 loss 权重开启也没有有效 pair | 暂标为未接通；若启用，需要 manifest 持久化 pair_id 并让 trainer 读取 |
| synthetic horizon augmentation | 配置存在，训练消费 manifest，但当前文档未说明生成/缓存/DP 重跑验收 | 保留为 ablation/预处理扩展，不作为 Phase I 默认链路 |
| local optimum escape | `CodebookLocalOptimumEscapeConfig` 与字段文档存在，但 trainer 未调用扰动逻辑 | 标为预留机制；不要写入已实现防护 |
| TensorBoard / latent snapshots | diagnostics 模块存在，trainer 当前只写 JSON diagnostics | 输出产物表已标记为规划/未接入 |
| failure case report | `FailureCaseReportBuilder` 存在，trainer 当前未调用 | 若要作为验收产物，需要从 `per_horizon_replay_records` 接入 |
| early stopping | `TrainingConfig.early_stopping_patience` 存在，trainer 未实现 early stopping | 配置应标为预留，或补 trainer 逻辑 |
| evaluation cadence 配置化 | trainer 内部 `evaluation_interval = 5` 硬编码；`full_validation_every_epochs` 只决定 evaluated epoch 是 full 还是 fast | 若希望 CLI 可控，需要新增 `validation_every_epochs` 或复用现有字段 |

### 13.3 仍建议补充到后续设计/实现的验收

- 增加一条配置一致性测试：设计文档中的关键默认值必须与 `Phase1Config` / `Phase1DataProcessConfig` 一致，尤其是 `beta0`、`commission_rate`、`reward_normalization`、`reward_clip_value`、`ema_decay`。
- 增加论文复现实验模板：固定 `--paper-strict-reproduction`、`commission_rate=0.0002`、`num_demos=30000`、`horizon=72`、`K=10`、`code_dim=16`、`hidden_dim=128`、`epochs=100`，并在报告里写入 strict overrides。
- 增加 no-trade 语义检查：分别报告 `no_trade_ratio`、`active_trade_code_count`、`no_trade_code_concentration_top1/top2`，明确工程数据集不是论文文字里“每个 horizon 必有一次交易”的严格子集。
- 增加 reward alignment 签收检查：`paper_formula` 与 `next_row_execution` 的实验不可混合比较；Phase II 必须继承 Phase I 的 `reward_alignment`、`commission_rate`、`max_position`。

## 14. 测试计划

### 14.1 单元测试

```text
tests/unit/config/test_phase1_config_docs.py
tests/unit/data/test_dataset.py
tests/unit/data/test_state_normalizer.py
tests/unit/models/test_vq_archetype.py
tests/unit/models/test_vector_quantizer.py
tests/unit/models/test_vq_losses.py
tests/unit/models/test_encoder_inputs.py
tests/unit/models/test_reward_normalizer.py
tests/unit/trainers/test_phase1_trainer.py
tests/unit/trainers/test_phase1_checkpoint.py
tests/unit/trainers/test_phase1_selection_policy.py
tests/unit/evaluation/test_phase1_evaluator.py
tests/unit/evaluation/test_phase1_metrics.py
tests/unit/evaluation/test_phase1_replay.py
tests/unit/evaluation/test_phase1_report.py
tests/unit/trading/test_env.py
tests/unit/trading/test_cost_model.py
tests/unit/trading/test_reward_alignment.py
```

### 14.2 关键单元测试用例

| 测试 | 关键不变量 |
| --- | --- |
| `test_decoder_causal_masking` | 修改 `s_{τ+1:}` 不改变 `logits[:τ+1]` |
| `test_decoder_bidirectional_false` | LSTM `bidirectional=False` |
| `test_vector_quantizer_ste_gradient` | STE 梯度正确传播 |
| `test_codebook_ema_update` | EMA 更新后 codebook 值与手算一致 |
| `test_dead_code_restart_resets_code` | restart 后 dead code 被重置 |
| `test_reward_normalizer_robust_fallback` | kurtosis 低于阈值时回退 standard |
| `test_state_normalizer_train_only_fit` | val/test 不参与 fit |
| `test_selection_policy_blocks_low_usage` | `code_usage_ratio < 0.7` → reject |
| `test_selection_policy_blocks_high_drawdown` | `max_drawdown > 0.2` → reject |
| `test_checkpoint_atomic_write` | 中途异常不留下半成品 |
| `test_trainer_fatal_collapse_exit_code` | 连续 collapse 时非零退出码 |
| `test_composite_score_sensitivity` | 权重扰动后 best epoch 漂移被检测 |

### 14.3 集成测试

```text
tests/integration/test_phase1_pipeline_smoke.py
tests/integration/test_phase1_data_process_then_train.py
tests/integration/test_phase1_next_row_alignment.py
tests/integration/test_phase1_collapse_handling.py
tests/integration/test_phase1_reproducibility.py
```

### 14.4 关键集成测试场景

| 测试 | 关键不变量 |
| --- | --- |
| `test_phase1_pipeline_smoke` | small fixture 上跑一轮完整训练 + 评估 + 产物导出 |
| `test_phase1_data_process_then_train` | manifest 模式下不重新枚举窗口、不重新运行 DP |
| `test_phase1_collapse_handling` | 连续 collapse 时训练以非零退出码结束 |
| `test_phase1_reproducibility` | 固定 seed + manifest 得到一致的 best checkpoint |

## 15. 命令示例

数据处理（前置阶段）：

```bash
python scripts/pre_process_data.py \
  --pair FU \
  --data-batch-id batch_001 \
  --train-file data/FU/train.feather \
  --val-file data/FU/val.feather \
  --test-file data/FU/test.feather \
  --factor-profile short \
  --horizon 72 \
  --num-demos 30000 \
  --reward-alignment paper_formula \
  --artifact-root artifacts
```

Phase I 训练：

```bash
python scripts/train_phase1.py \
  --pair FU \
  --train-batch-id phase1_vq_001 \
  --data-process-manifest artifacts/FU/batch_001/phase1/data_process_manifest.json \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.001 \
  --device cuda
```

严格论文复现：

```bash
python scripts/train_phase1.py \
  --pair FU \
  --train-batch-id phase1_vq_strict_001 \
  --data-process-manifest artifacts/FU/batch_001/phase1/data_process_manifest.json \
  --epochs 100 \
  --paper-strict-reproduction
```
