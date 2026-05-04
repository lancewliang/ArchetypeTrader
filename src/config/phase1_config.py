"""Phase I 集中配置。

设计文档锚点: §4.2 与各章节的 yaml 默认值。

设计原则:
- 配置必须可序列化为 yaml/json（写入 ``phase1_config.yaml`` / ``phase1_report.json``）。
- 所有默认值必须与设计文档一致；偏离时应在设计中明确记录。
- 模型行为开关（如 ``train_reward_robust``、``dead_code_restart`` 默认 true、
  ``require_prospective_diagnostic`` 默认 true）锁在本文件，避免训练入口偷偷改变。

实现要点:
- 顶层 ``Phase1Config`` 是 frozen dataclass + 嵌套 frozen dataclass，保证不可变；
  任何"运行时偷改"都需要通过 ``dataclasses.replace`` 明确生成新实例。
- ``config_hash`` 必须稳定: 相同字段 → 相同 hash；用于 cache 失效检测。
- ``paper_strict_reproduction=True`` 时调用 ``apply_paper_strict_overrides`` 自动
  关闭工程稳定项（usage_regularization / dead_code_restart / robust normalization），
  使训练严格对齐论文公式 (4)。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


# ---------- 数据 / 采样 ----------

@dataclass(frozen=True)
class StratificationConfig:
    """分层采样配置。

    强制 prospective 对照: ``require_prospective_diagnostic`` 默认 ``True``，
    主实验在缺少诊断 BATCH_ID 时不可启动。
    """
    mode: Literal["hindsight_horizon", "prospective_past"] = "hindsight_horizon"
    prospective_lookback_minutes: int = 1440
    require_prospective_diagnostic: bool = True
    diagnostic_pair_batch_id: Optional[str] = None
    report_hindsight_bias_warning: bool = True
    hindsight_vs_prospective_max_delta: Dict[str, float] = field(
        default_factory=lambda: {
            "val_return_capture_ratio": 0.20,
            "val_sharpe_ratio": 0.50,
            "val_max_drawdown": 0.10,
            "code_usage_ratio": 0.10,
        }
    )


@dataclass(frozen=True)
class SamplingHealthConfig:
    """采样健康阈值。

    ``split_boundary_embargo`` 必须同时覆盖 markout 行:
    - ``paper_formula``: ``h + 1``
    - ``next_row_execution``: ``h + 2``
    """
    max_no_trade_ratio: float = 0.25
    flat_low_vol_max_ratio: float = 0.15
    min_gap_between_samples: int = 12  # h=72 时的 50% overlap 上限
    max_overlap_ratio: float = 0.5
    split_boundary_embargo: int = 73            # paper_formula 默认 (h + 1)
    next_row_split_boundary_embargo: int = 74   # next_row_execution 默认 (h + 2)
    warn_only: bool = False
    allow_overlap_relaxation: bool = False


@dataclass(frozen=True)
class NoTradeControlConfig:
    """No-trade 样本处理。"""
    keep_no_trade: bool = True
    max_no_trade_ratio: float = 0.35
    min_no_trade_ratio: float = 0.10
    min_low_opportunity_ratio: float = 0.25
    low_opportunity_return_quantile: float = 0.30
    min_profit_gate: float = 0.0
    cap_flat_low_vol_strata: bool = True
    flat_low_vol_max_ratio: float = 0.15
    resample_when_exceeded: bool = True
    resample_when_below_min: bool = True


@dataclass(frozen=True)
class TimeDistributionSamplingConfig:
    """Train split 的完整时间分布采样配置。"""
    enabled: bool = True
    full_time_mode: Literal["non_overlap", "stride"] = "stride"
    full_time_stride: int = 36
    min_train_ratio: float = 0.40
    label_export_enabled: bool = True


@dataclass(frozen=True)
class EvalLabelingConfig:
    """Eval split 的 label 生成契约。"""
    val_mode: Literal["horizon_stride", "all_eligible"] = "horizon_stride"
    test_mode: Literal["horizon_stride", "all_eligible"] = "horizon_stride"
    apply_sampling: bool = False
    apply_augmentation: bool = False


@dataclass(frozen=True)
class NoTradeCodeHealthConfig:
    """No-trade archetype 容量监控阈值。"""
    max_per_code_no_trade_ratio: float = 0.8
    max_top2_no_trade_concentration: float = 0.7
    min_active_trade_code_count: int = 6


@dataclass(frozen=True)
class TemporalContrastiveConfig:
    enabled: bool = False
    shift_bars: List[int] = field(default_factory=lambda: [-2, -1, 1, 2])
    pair_ratio: float = 0.5
    max_pairs: int = 30000
    require_same_strata: bool = False
    rerun_dp_for_shifted: bool = True
    contrastive_weight: float = 0.05
    temperature: float = 0.1


@dataclass(frozen=True)
class SyntheticHorizonConfig:
    enabled: bool = False
    synthetic_ratio: float = 0.1
    max_synthetic_horizons: int = 3000
    source_selection: str = "contrasting_strata"
    blend_window: int = 8
    min_source_distance: int = 72
    require_orderbook_consistency: bool = True
    rerun_dp: bool = True
    exclude_from_validation: bool = True


@dataclass(frozen=True)
class DataAugmentationConfig:
    temporal_contrastive: TemporalContrastiveConfig = field(
        default_factory=TemporalContrastiveConfig
    )
    synthetic_horizon: SyntheticHorizonConfig = field(
        default_factory=SyntheticHorizonConfig
    )


# ---------- DP / 成本 / reward 对齐 ----------

@dataclass(frozen=True)
class RejectTransitionHealthConfig:
    """盘口深度不足导致的转移拒绝监控。"""
    max_horizon_reject_rate: float = 0.10
    max_dataset_reject_rate: float = 0.05
    fail_when_exceeded: bool = True


@dataclass(frozen=True)
class CostConfig:
    """统一交易成本配置（DP teacher / student replay 必须共用）。"""
    reward_alignment: Literal["paper_formula", "next_row_execution"] = "paper_formula"
    commission_rate: float = 0.0005  # 论文 δ = 0.02%
    slippage_model: Literal["lob_depth"] = "lob_depth"
    book_levels: int = 5
    mark_price: Literal["mid_price"] = "mid_price"
    execution_lag: int = 0  # paper_formula 必须为 0
    insufficient_depth_policy: Literal["reject_transition"] = "reject_transition"
    reject_transition_health: RejectTransitionHealthConfig = field(
        default_factory=RejectTransitionHealthConfig
    )


@dataclass(frozen=True)
class DPConfig:
    """Single-trade DP 配置。"""
    horizon: int = 72
    gamma: float = 1.0
    max_position: int = 1
    cost_config: CostConfig = field(default_factory=CostConfig)


# ---------- 模型 ----------

@dataclass(frozen=True)
class EncoderInputConfig:
    """Encoder 输入适配。

    ``reward_normalization`` 默认 ``train_reward_robust``，配合 ``reward_clip_value=8.0``
    避免重尾分布把大行情切换点信号剪掉。
    """
    state_adapter_dim: int = 96
    action_embedding_dim: int = 16
    reward_embedding_dim: int = 16
    fusion_dim: int = 128
    reward_normalization: Literal[
        "train_reward_robust", "train_reward_standard"
    ] = "train_reward_standard"
    reward_clip_value: float = 5.0
    fallback_to_standard_kurtosis_below: float = 6.0


@dataclass(frozen=True)
class CodebookLocalOptimumEscapeConfig:
    enabled: bool = False
    stagnant_usage_epochs: int = 8
    stagnant_perplexity_epochs: int = 8
    best_score_no_improve_epochs: int = 15
    perturbation_probability_per_epoch: float = 0.05
    perturbation_std_ratio: float = 0.01
    min_epochs_between_perturbations: int = 10
    only_perturb_low_usage_codes: bool = True


@dataclass(frozen=True)
class CodebookHealthConfig:
    """Codebook collapse 防护。

    默认 ``dead_code_restart=True``；严格复现论文公式 (4) 时通过
    ``Phase1Config.training.paper_strict_reproduction`` 关闭 usage regularization 与 restart。
    """
    min_code_usage_ratio: float = 0.5
    max_dominant_code_ratio: float = 0.5
    usage_regularization_weight: float = 0.01
    usage_profit_alignment_weight: float = 0.05
    usage_profit_alignment_target_corr: float = 0.2
    usage_profit_alignment_temperature: float = 2.0
    dead_code_patience: int = 5
    dead_code_restart: bool = True
    restart_source: str = "high_reconstruction_error_samples"
    restart_cooldown_epochs: int = 3
    consecutive_collapse_epoch_limit: int = 10
    local_optimum_escape: CodebookLocalOptimumEscapeConfig = field(
        default_factory=CodebookLocalOptimumEscapeConfig
    )


@dataclass(frozen=True)
class CodebookConfig:
    init_method: Literal[
        "random_normal", "sample_encoder_outputs", "kmeans_warmup"
    ] = "kmeans_warmup"
    kmeans_warmup_batches: int = 64
    update_method: Literal["gradient", "ema"] = "ema"
    ema_decay: float = 0.95
    ema_epsilon: float = 1.0e-5
    gumbel_temperature: float = 2.0
    health: CodebookHealthConfig = field(default_factory=CodebookHealthConfig)


@dataclass(frozen=True)
class ModelConfig:
    """VQ encoder-decoder 整体超参。"""
    hidden_dim: int = 128
    code_dim: int = 16
    num_codes: int = 10
    beta0: float = 0.25
    encoder_input: EncoderInputConfig = field(default_factory=EncoderInputConfig)
    codebook: CodebookConfig = field(default_factory=CodebookConfig)


# ---------- 训练 / 选择 / 评估 ----------

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 256
    lr: float = 1.0e-3
    epochs: int = 100
    pretrain_epochs: int = 10
    pretrain_lr: Optional[float] = None
    seed: int = 42
    device: str = "cuda"
    save_every: int = 10
    early_stopping_patience: Optional[int] = None
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    paper_strict_reproduction: bool = False  # True 时关闭工程稳定项以严格复现论文
    full_validation_every_epochs: int = 5
    fast_val_probe_size: int = 2048


@dataclass(frozen=True)
class RiskGuardrailConfig:
    max_drawdown: float = 0.2
    min_sharpe_ratio: float = 0.0


@dataclass(frozen=True)
class BehaviorGuardrailConfig:
    min_inter_code_action_diversity: float = 0.15
    min_decoder_sensitivity_to_code: float = 0.05
    min_epoch_code_stability: float = 0.8


@dataclass(frozen=True)
class TeacherQualityGuardrailConfig:
    min_dp_teacher_profitable_ratio: float = 0.3


@dataclass(frozen=True)
class SelectionPolicyConfig:
    """Best checkpoint 选择策略。

    集中所有 guardrail，便于审计；evaluator 不知道任何阈值。
    """
    selection_metric: str = "phase1_composite_score"
    selection_mode: Literal["max", "min"] = "max"
    min_code_usage_ratio: float = 0.7
    metric_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "switch_point_recall": 0.30,
            "switch_direction_accuracy": 0.20,
            "val_weighted_reconstruction_accuracy": 0.20,
            "val_return_capture_ratio": 0.20,
            "val_sharpe_ratio": 0.10,
        }
    )
    risk: RiskGuardrailConfig = field(default_factory=RiskGuardrailConfig)
    behavior: BehaviorGuardrailConfig = field(default_factory=BehaviorGuardrailConfig)
    teacher: TeacherQualityGuardrailConfig = field(
        default_factory=TeacherQualityGuardrailConfig
    )
    composite_score_sensitivity_perturbations: List[Dict[str, float]] = field(
        default_factory=lambda: [
            {"switch_point_recall": +0.10},
            {"switch_point_recall": -0.10},
            {"val_return_capture_ratio": +0.10},
            {"val_return_capture_ratio": -0.10},
            {"val_sharpe_ratio": +0.05},
            {"val_sharpe_ratio": -0.05},
        ]
    )


@dataclass(frozen=True)
class DiagnosticsConfig:
    tensorboard_enabled: bool = True
    tensorboard_log_every_epochs: int = 5
    tensorboard_max_points_per_split: int = 2000
    fixed_probe_seed: int = 2026
    latent_visualization_enabled: bool = True
    latent_projections: List[str] = field(default_factory=lambda: ["pca", "tsne"])
    failure_cases_enabled: bool = True
    failure_cases_top_k: int = 10


# ---------- 顶层配置 ----------

@dataclass(frozen=True)
class Phase1DataProcessConfig:
    """Offline Phase I data processing config.

    It contains only fields that can affect sampled horizons or DP teacher
    outputs. Model and optimizer fields intentionally stay in ``Phase1Config``.
    """
    pair: str
    data_batch_id: str
    train_file: str
    val_file: str
    test_file: str
    artifact_root: str = "artifacts"
    factor_profile: str = "short"
    factor_list_file: Optional[str] = None
    horizon: int = 72
    num_demos: int = 30000
    sampling_strategy: Literal[
        "stratified_uniform", "stratified_proportional"
    ] = "stratified_uniform"
    stratification: StratificationConfig = field(default_factory=StratificationConfig)
    sampling_health: SamplingHealthConfig = field(default_factory=SamplingHealthConfig)
    no_trade_control: NoTradeControlConfig = field(
        default_factory=NoTradeControlConfig
    )
    time_distribution_sampling: TimeDistributionSamplingConfig = field(
        default_factory=TimeDistributionSamplingConfig
    )
    eval_labeling: EvalLabelingConfig = field(default_factory=EvalLabelingConfig)
    data_augmentation: DataAugmentationConfig = field(
        default_factory=DataAugmentationConfig
    )
    dp: DPConfig = field(default_factory=DPConfig)
    dp_workers: int = 0  # 0=auto; large datasets use available CPU cores
    dp_worker_chunksize: int = 32
    seed: int = 42
    allow_missing_prospective_diagnostic: bool = False
    risk_acknowledged_by: Optional[str] = None
    expected_sign_off_followup_batch_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def artifacts_dir(self) -> Path:
        return Path(self.artifact_root) / self.pair / self.data_batch_id / "phase1"


@dataclass(frozen=True)
class Phase1Config:
    """Phase I 训练配置。

    仅包含训练阶段所需字段。数据预处理字段已迁移至 ``Phase1DataProcessConfig``。
    训练必须通过 ``data_process_manifest`` 指向离线数据预处理产物。
    """
    pair: str
    train_batch_id: str
    data_process_manifest: str
    artifact_root: str = "artifacts"
    horizon: int = 72
    dp: DPConfig = field(default_factory=DPConfig)
    stratification: StratificationConfig = field(default_factory=StratificationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    selection_policy: SelectionPolicyConfig = field(
        default_factory=SelectionPolicyConfig
    )
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    data_augmentation: DataAugmentationConfig = field(
        default_factory=DataAugmentationConfig
    )
    local_smoke_relaxed_guardrails: bool = False
    factor_profile: str = "short"
    factor_list_file: Optional[str] = None
    allow_missing_prospective_diagnostic: bool = False
    risk_acknowledged_by: Optional[str] = None
    expected_sign_off_followup_batch_id: Optional[str] = None

    # ---- 序列化与 hash ----

    def to_dict(self) -> dict:
        """转 dict 以便 yaml/json 序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Phase1Config":
        """从 dict 重建（对应 ``to_dict()`` 的逆操作，主要给单测/反序列化用）。

        因为 ``from __future__ import annotations`` 下 ``dataclass.fields`` 的
        ``f.type`` 是字符串而不是类对象，无法靠反射递归还原嵌套 dataclass。
        这里维护一个显式 ``nested_fields`` 映射，对每个嵌套字段调用
        ``_rebuild_nested``（递归到 ``_NESTED_TYPE_MAP``）。
        """
        # 复制一份，避免修改调用方传入的 dict。
        data = dict(payload)
        nested_fields = {
            "stratification": StratificationConfig,
            "dp": DPConfig,
            "model": ModelConfig,
            "training": TrainingConfig,
            "selection_policy": SelectionPolicyConfig,
            "diagnostics": DiagnosticsConfig,
            "data_augmentation": DataAugmentationConfig,
        }
        for name, klass in nested_fields.items():
            if name in data and isinstance(data[name], dict):
                data[name] = _rebuild_nested(klass, data[name])
        return cls(**data)

    def write_yaml(self, path: Path) -> Path:
        """原子写 ``phase1_config.yaml``，便于复现实验。"""
        import yaml

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # yaml 的原子写: 复用 atomic_write_json 不合适（json 与 yaml 行为不同），
        # 这里用同样思路: 写 .tmp + os.replace。
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=target.parent,
            suffix=".tmp",
            encoding="utf-8",
        ) as tmp:
            yaml.safe_dump(
                self.to_dict(),
                tmp,
                allow_unicode=True,
                sort_keys=True,
                default_flow_style=False,
            )
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, target)
        return target

    def config_hash(self) -> str:
        """稳定 hash: 相同字段 → 相同 hash。

        实现思路: ``json.dumps(to_dict(), sort_keys=True, separators=(",", ":"))``
        再取 sha256 前 16 位。
        """
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def training_config_hash(self) -> str:
        """训练配置 hash: 只覆盖影响模型训练行为的字段。

        不包含标识符（pair / train_batch_id / artifact_root / data_process_manifest）
        和审计字段（allow_missing_prospective_diagnostic / risk_acknowledged_by /
        expected_sign_off_followup_batch_id / factor_profile / factor_list_file），
        这些不改变训练行为。
        """
        training_only = {
            "horizon": self.horizon,
            "dp": asdict(self.dp),
            "stratification": asdict(self.stratification),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "selection_policy": asdict(self.selection_policy),
            "diagnostics": asdict(self.diagnostics),
            "data_augmentation": asdict(self.data_augmentation),
            "local_smoke_relaxed_guardrails": self.local_smoke_relaxed_guardrails,
        }
        canonical = json.dumps(
            training_only,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    # ---- 输出目录 ----

    def artifacts_dir(self) -> Path:
        """``artifacts/{pair}/{train_batch_id}/phase1/``。"""
        return Path(self.artifact_root) / self.pair / self.train_batch_id / "phase1"


def _rebuild_nested(klass, payload: dict):
    """递归把 dict 还原为嵌套 dataclass。

    封装在外层避免污染 Phase1Config 的命名空间。
    """
    kwargs = {}
    for fname, finfo in klass.__dataclass_fields__.items():
        if fname not in payload:
            continue
        value = payload[fname]
        # 检查 default_factory 推导嵌套类型；这里采用更直接的策略: 调用方知道映射。
        if isinstance(value, dict):
            # 尝试找子类型
            sub = _NESTED_TYPE_MAP.get((klass.__name__, fname))
            if sub is not None:
                value = _rebuild_nested(sub, value)
        kwargs[fname] = value
    return klass(**kwargs)


_NESTED_TYPE_MAP: Dict[tuple, type] = {
    ("DataAugmentationConfig", "temporal_contrastive"): TemporalContrastiveConfig,
    ("DataAugmentationConfig", "synthetic_horizon"): SyntheticHorizonConfig,
    ("DPConfig", "cost_config"): CostConfig,
    ("CostConfig", "reject_transition_health"): RejectTransitionHealthConfig,
    ("ModelConfig", "encoder_input"): EncoderInputConfig,
    ("ModelConfig", "codebook"): CodebookConfig,
    ("CodebookConfig", "health"): CodebookHealthConfig,
    ("CodebookHealthConfig", "local_optimum_escape"): CodebookLocalOptimumEscapeConfig,
    ("SelectionPolicyConfig", "risk"): RiskGuardrailConfig,
    ("SelectionPolicyConfig", "behavior"): BehaviorGuardrailConfig,
    ("SelectionPolicyConfig", "teacher"): TeacherQualityGuardrailConfig,
}


def _config_doc(why: str, tuning_effect: str) -> Dict[str, str]:
    """构造字段说明；用于生成配置文档和 IDE 检索。"""
    return {"why": why, "tuning_effect": tuning_effect}


# ---------- 配置字段说明 ----------

PHASE1_CONFIG_FIELD_DOCS: Dict[str, Dict[str, str]] = {
    "pair": _config_doc(
        "标识本次训练对应的交易品种，用于定位输入数据和产物目录。",
        "换品种时必须调整；同一 batch 内不应混用不同品种，否则数据、报告和缓存会不可追溯。",
    ),
    "train_batch_id": _config_doc(
        "标识一次 Phase I 实验批次，用于隔离产物、报告和缓存。",
        "改大或改名会写入新目录，适合 ablation；复跑同一实验应保持不变以便校验 hash。",
    ),
    "data_process_manifest": _config_doc(
        "Phase I 训练的唯一强制入口，指向离线数据预处理产物清单。",
        "必须由 ``scripts/phase1_data_processor.py`` 生成；训练只读取已固化的 sampled horizons 与 DP teacher。",
    ),
    "artifact_root": _config_doc(
        "配置 Phase I 产物根目录，集中保存配置、schema、demo、模型和报告。",
        "调整后会改变输出位置；不改变模型行为，但会影响后续 Phase II 查找产物。",
    ),
    "horizon": _config_doc(
        "定义每个 demonstration 的固定时间长度，也是 DP 和 decoder 的序列长度。",
        "增大可覆盖更长交易机会但计算更重、重叠风险更高；减小会偏向短周期模式。",
    ),
    "stratification": _config_doc(
        "集中控制后视/前瞻分层和 hindsight bias 诊断。",
        "调整子项会改变样本选择边界和报告签收条件，是比较实验必须固定的核心配置组。",
    ),
    "dp": _config_doc(
        "集中控制 single-trade DP teacher 的 horizon、折扣、仓位和成本语义。",
        "改动会直接改变 demonstration actions/rewards；旧 DP 缓存必须失效重算。",
    ),
    "model": _config_doc(
        "配置 VQ encoder-decoder 的容量、codebook 和输入适配。",
        "增大容量提高表达力但更容易过拟合或 collapse；减小更稳但可能欠拟合。",
    ),
    "training": _config_doc(
        "配置优化过程、设备、复现 seed 和 checkpoint 频率。",
        "影响收敛速度、稳定性和可复现性；实验对比时应尽量固定。",
    ),
    "selection_policy": _config_doc(
        "配置 best checkpoint 选择指标和 guardrail。",
        "改权重会改变被提升的 checkpoint；改阈值会改变哪些模型允许签收。",
    ),
    "diagnostics": _config_doc(
        "配置 TensorBoard、latent 可视化和失败样本输出。",
        "开启更多诊断便于审计但增加耗时和产物体积；关闭只适合快速本地验证。",
    ),
    "allow_missing_prospective_diagnostic": _config_doc(
        "紧急情况下允许缺少前瞻分层对照实验继续运行。",
        "设为 true 会降低签收可信度，必须配合风险确认字段；正式实验应保持 false。",
    ),
    "risk_acknowledged_by": _config_doc(
        "记录谁确认接受缺少前瞻诊断或其他采样风险。",
        "仅在豁免路径填写；为空时表示没有人工风险豁免。",
    ),
    "expected_sign_off_followup_batch_id": _config_doc(
        "记录后续补跑的签收批次，避免临时豁免变成永久结论。",
        "填写后可在报告中追踪 follow-up；为空表示无计划或不需要补跑。",
    ),
    "local_smoke_relaxed_guardrails": _config_doc(
        "允许本地 smoke test 放宽 guardrail，避免小 fixture 被正式阈值拦住。",
        "仅用于本地/CI 轻量验证；正式训练应保持 false，防止低质量模型被签收。",
    ),
    "factor_profile": _config_doc(
        "选择内置因子字段集合，用于报告审计。",
        "应与数据预处理阶段使用的 profile 一致；实际特征列由 manifest 中的 schema 决定。",
    ),
    "factor_list_file": _config_doc(
        "外部因子清单路径，用于报告审计。",
        "应与数据预处理阶段使用的因子清单一致；实际特征列由 manifest 中的 schema 决定。",
    ),

    "stratification.mode": _config_doc(
        "选择用 horizon 内统计还是过去窗口统计做分层，隔离 hindsight bias。",
        "hindsight_horizon 覆盖更均衡但带后视选择风险；prospective_past 更接近线上可观测条件。",
    ),
    "stratification.prospective_lookback_minutes": _config_doc(
        "定义前瞻诊断分层使用的过去观察窗口长度。",
        "增大更平滑但反应慢；减小更灵敏但 strata 噪声更大。",
    ),
    "stratification.require_prospective_diagnostic": _config_doc(
        "强制主实验必须配套前瞻分层诊断，防止只报告后视采样收益。",
        "true 提高签收门槛；false 只适合研究中间态或明确豁免场景。",
    ),
    "stratification.diagnostic_pair_batch_id": _config_doc(
        "指向配套前瞻诊断批次，用于读取报告并比较核心指标。",
        "填写后可自动做 hindsight/prospective 差异检查；为空时主实验可能无法签收。",
    ),
    "stratification.report_hindsight_bias_warning": _config_doc(
        "控制是否在报告中显式写出后视分层风险。",
        "保持 true 便于审计；关闭会减少报告噪声但不建议用于正式实验。",
    ),
    "stratification.hindsight_vs_prospective_max_delta": _config_doc(
        "定义后视实验相对前瞻诊断的核心指标差异上限。",
        "阈值越小越保守；阈值越大更容易通过但隐藏后视采样收益的风险更高。",
    ),

    "data_augmentation": _config_doc(
        "集中控制训练集增强，默认关闭以保持论文主实验干净。",
        "开启可增加鲁棒性和样本量，但会改变训练分布，必须使用独立 batch 做 ablation。",
    ),
    "data_augmentation.temporal_contrastive": _config_doc(
        "配置时序偏移对比学习，帮助 encoder 对轻微时间错位保持稳健。",
        "开启会增加辅助 loss 和训练成本；关闭保持论文主实验更干净。",
    ),
    "data_augmentation.temporal_contrastive.enabled": _config_doc(
        "控制是否生成 temporal contrastive pair 并加入对比损失。",
        "true 可能提升 latent 稳定性；false 避免改变论文公式和训练目标。",
    ),
    "data_augmentation.temporal_contrastive.shift_bars": _config_doc(
        "定义构造对比 pair 的前后平移 bar 数。",
        "平移越大增强越强但语义可能变；平移越小更安全但收益有限。",
    ),
    "data_augmentation.temporal_contrastive.pair_ratio": _config_doc(
        "控制训练样本中生成对比 pair 的比例。",
        "提高会增强对比信号但挤占主重构训练；降低则影响更轻。",
    ),
    "data_augmentation.temporal_contrastive.max_pairs": _config_doc(
        "限制最多生成的对比 pair 数量，控制内存和训练时间。",
        "增大覆盖更多样本但更耗资源；减小适合快速实验。",
    ),
    "data_augmentation.temporal_contrastive.require_same_strata": _config_doc(
        "要求对比 pair 保持同一 strata，减少语义错配。",
        "true 更保守但可生成 pair 更少；false 覆盖更广但噪声可能增加。",
    ),
    "data_augmentation.temporal_contrastive.rerun_dp_for_shifted": _config_doc(
        "控制平移后的 horizon 是否重新跑 DP，而不是复用原动作。",
        "true 更准确但更慢；false 更快但可能让动作标签和窗口错位。",
    ),
    "data_augmentation.temporal_contrastive.contrastive_weight": _config_doc(
        "设置对比学习 loss 在总损失中的权重。",
        "增大会更强约束 latent 稳定性；过大可能牺牲动作重构。",
    ),
    "data_augmentation.temporal_contrastive.temperature": _config_doc(
        "控制对比学习 softmax 温度，影响正负样本分离强度。",
        "降低会让分离更尖锐但训练不稳；提高更平滑但约束变弱。",
    ),
    "data_augmentation.synthetic_horizon": _config_doc(
        "配置合成 horizon 扩充，只允许用于 train split。",
        "开启可扩充稀有模式；关闭避免合成分布影响正式基线。",
    ),
    "data_augmentation.synthetic_horizon.enabled": _config_doc(
        "控制是否生成 synthetic horizon。",
        "true 增加训练样本和覆盖；false 保持真实数据分布。",
    ),
    "data_augmentation.synthetic_horizon.synthetic_ratio": _config_doc(
        "设置合成样本相对真实样本的比例。",
        "增大增强更强但分布偏移更大；减小影响更可控。",
    ),
    "data_augmentation.synthetic_horizon.max_synthetic_horizons": _config_doc(
        "限制合成 horizon 总数，避免增强无限膨胀。",
        "增大覆盖更多组合但更耗时；减小适合诊断或小显存环境。",
    ),
    "data_augmentation.synthetic_horizon.source_selection": _config_doc(
        "定义合成样本来源窗口的选择策略。",
        "contrasting_strata 强化差异模式；换策略会改变合成样本语义和分布。",
    ),
    "data_augmentation.synthetic_horizon.blend_window": _config_doc(
        "控制合成窗口拼接或混合的过渡长度。",
        "增大过渡更平滑但可能抹掉突变；减小保留突变但更容易不自然。",
    ),
    "data_augmentation.synthetic_horizon.min_source_distance": _config_doc(
        "要求合成来源窗口之间至少相隔多少 bar，降低近邻重复。",
        "增大去相关更强但候选更少；减小更容易生成但相似度更高。",
    ),
    "data_augmentation.synthetic_horizon.require_orderbook_consistency": _config_doc(
        "要求合成后的盘口字段保持成交语义一致。",
        "true 更安全但可合成样本更少；false 风险较高，可能制造无法成交的样本。",
    ),
    "data_augmentation.synthetic_horizon.rerun_dp": _config_doc(
        "控制合成 horizon 是否重新跑 DP 获取动作和 reward。",
        "true 保证标签匹配合成价格/盘口；false 更快但标签可信度低。",
    ),
    "data_augmentation.synthetic_horizon.exclude_from_validation": _config_doc(
        "确保合成样本不会进入 validation/test 指标。",
        "正式实验应保持 true；false 会污染泛化评估，通常只用于调试。",
    ),

    "dp.horizon": _config_doc(
        "DP teacher 使用的序列长度，应与顶层 horizon 保持一致。",
        "增减会改变 DP 状态空间和动作标签；不一致会导致训练样本 shape 错误。",
    ),
    "dp.gamma": _config_doc(
        "DP 累积收益折扣因子，控制远期 reward 权重。",
        "低于 1 会偏向更早收益；等于 1 与论文单 horizon 累积收益更一致。",
    ),
    "dp.max_position": _config_doc(
        "定义 action 映射后的最大持仓规模。",
        "增大会放大收益、成本和盘口深度约束；减小更保守。",
    ),
    "dp.cost_config": _config_doc(
        "配置 DP、replay 和评估共用的成本与 reward 对齐语义。",
        "任何改动都会改变 rewards 和可成交性，必须让旧缓存失效。",
    ),
    "dp.cost_config.reward_alignment": _config_doc(
        "定义动作成交行和 markout 行的时间对齐方式。",
        "paper_formula 便于论文复现；next_row_execution 更保守但不能直接和论文指标比较。",
    ),
    "dp.cost_config.commission_rate": _config_doc(
        "设置换仓手续费率，反映交易成本。",
        "提高会减少 DP 交易频率和收益；降低会让策略更愿意切换仓位。",
    ),
    "dp.cost_config.slippage_model": _config_doc(
        "选择滑点模型，当前默认按盘口深度估算真实成交成本。",
        "lob_depth 更贴近成交约束；若未来增加模型，切换会改变 teacher/replay 可比性。",
    ),
    "dp.cost_config.book_levels": _config_doc(
        "指定盘口深度滑点使用的档位数。",
        "增大可容纳更大仓位但需要更多字段；减小更保守且更容易 reject transition。",
    ),
    "dp.cost_config.mark_price": _config_doc(
        "定义 reward 结算使用的 mark price 类型。",
        "mid_price 减少买卖价噪声；切换口径会改变收益尺度和评估可比性。",
    ),
    "dp.cost_config.execution_lag": _config_doc(
        "表达观察到成交之间的延迟，paper_formula 下固定为 0。",
        "next_row_execution 可设为 1 表示下一行成交；错误调整会造成 reward 行号错位。",
    ),
    "dp.cost_config.insufficient_depth_policy": _config_doc(
        "定义盘口深度不足时如何处理换仓转移。",
        "reject_transition 更真实但可能提高 no-trade；放松策略会增加成交假设风险。",
    ),
    "dp.cost_config.reject_transition_health": _config_doc(
        "监控盘口深度不足导致的 DP 转移拒绝率。",
        "阈值越严越早暴露数据/流动性问题；越松更容易训练但 demo 质量可能下降。",
    ),
    "dp.cost_config.reject_transition_health.max_horizon_reject_rate": _config_doc(
        "限制单个 horizon 内被拒绝转移的最高比例。",
        "降低会更快定位异常窗口；提高会容忍局部流动性不足。",
    ),
    "dp.cost_config.reject_transition_health.max_dataset_reject_rate": _config_doc(
        "限制整个数据集层面的转移拒绝比例。",
        "降低更保守；提高可容忍整体盘口偏稀疏的数据。",
    ),
    "dp.cost_config.reject_transition_health.fail_when_exceeded": _config_doc(
        "控制拒绝率超阈值时是否阻止后续训练。",
        "true 防止低质量 demo 进入训练；false 只适合调试或诊断数据问题。",
    ),

    # 模型
    "model.hidden_dim": _config_doc(
        "设置 encoder/decoder 主干隐藏维度，控制模型容量。",
        "增大表达力更强但更慢、更易过拟合；减小更快但可能欠拟合。",
    ),
    "model.code_dim": _config_doc(
        "设置 VQ latent/codebook embedding 维度。",
        "增大可表达更细模式但 codebook 更难稳定；减小更稳但信息瓶颈更强。",
    ),
    "model.num_codes": _config_doc(
        "定义 archetype 数量 K。",
        "增大可发现更多策略类型但 dead code 风险更高；减小更稳但策略粒度更粗。",
    ),
    "model.beta0": _config_doc(
        "设置 commitment loss 权重，约束 encoder 输出贴近选中 code。",
        "增大会让 latent 更贴近 code 但可能限制表达；减小更自由但 code assignment 更漂。",
    ),
    "model.encoder_input": _config_doc(
        "配置 state/action/reward 输入进入 encoder 前的适配方式。",
        "调整会改变模态权重和 reward 尺度，是影响 latent 质量的关键入口。",
    ),
    "model.encoder_input.state_adapter_dim": _config_doc(
        "设置状态特征投影后的维度。",
        "增大保留更多状态信息但更耗算力；减小可正则化但可能丢失细节。",
    ),
    "model.encoder_input.action_embedding_dim": _config_doc(
        "设置 demonstration action 的 embedding 维度。",
        "增大让动作模式更容易被编码；过大可能盖过状态和 reward 信号。",
    ),
    "model.encoder_input.reward_embedding_dim": _config_doc(
        "设置逐步 reward 的 embedding 维度。",
        "增大强化收益形状信息；过大可能让 encoder 过度依赖 teacher reward。",
    ),
    "model.encoder_input.fusion_dim": _config_doc(
        "设置 state/action/reward 融合后的维度。",
        "增大提升融合容量；减小提高正则化和速度。",
    ),
    "model.encoder_input.reward_normalization": _config_doc(
        "定义 reward 输入归一化方式，控制重尾收益对 encoder 的影响。",
        "robust 更抗极端值；standard 更贴近常规标准化和严格复现实验。",
    ),
    "model.encoder_input.reward_clip_value": _config_doc(
        "设置归一化 reward 的裁剪边界。",
        "增大保留极端行情但可能不稳；减小更稳但可能剪掉关键切换信号。",
    ),
    "model.encoder_input.fallback_to_standard_kurtosis_below": _config_doc(
        "当训练 reward 分布接近正态时允许回退到 standard normalization。",
        "阈值越高越容易回退；阈值越低越偏向保持 robust。",
    ),
    "model.codebook": _config_doc(
        "配置 VQ codebook 初始化、更新和健康防护。",
        "调整会影响 code 使用率、collapse 风险和论文复现严格性。",
    ),
    "model.codebook.init_method": _config_doc(
        "定义 codebook 初始 embedding 的生成方式。",
        "kmeans_warmup 更稳；random_normal 更贴近简单复现但 dead code 风险高。",
    ),
    "model.codebook.kmeans_warmup_batches": _config_doc(
        "设置 K-means 初始化使用的 warmup batch 数。",
        "增大初始化更充分但更慢；减小更快但聚类代表性更弱。",
    ),
    "model.codebook.update_method": _config_doc(
        "定义 codebook embedding 更新方式。",
        "ema 更稳定；gradient 更贴近论文公式但训练早期更容易 collapse。",
    ),
    "model.codebook.ema_decay": _config_doc(
        "设置 EMA codebook 更新的衰减系数。",
        "增大更新更平滑但适应慢；减小适应快但 code 抖动更大。",
    ),
    "model.codebook.ema_epsilon": _config_doc(
        "EMA 更新时的数值稳定项，避免低计数 code 除零或爆炸。",
        "通常不需要调；过大可能偏置低使用率 code，过小可能数值不稳。",
    ),
    "model.codebook.gumbel_temperature": _config_doc(
        "Gumbel-Softmax 温度参数，控制多样性损失的梯度强度。增大使软分配更均匀（梯度更强但信号更模糊）；减小使软分配更尖锐（接近硬分配）。",
        "典型范围 0.1~10.0；collapse 严重时可增大到 5.0~10.0。",
    ),
    "model.codebook.health": _config_doc(
        "集中配置 codebook collapse 监控和恢复策略。",
        "阈值越严越能保证 archetype 可用性；过严可能让训练过早失败。",
    ),
    "model.codebook.health.min_code_usage_ratio": _config_doc(
        "要求被有效使用的 code 比例下限。",
        "提高会要求更多 archetype 活跃；降低可容忍较少模式但多样性下降。",
    ),
    "model.codebook.health.max_dominant_code_ratio": _config_doc(
        "限制单个 code 占据样本的最大比例。",
        "降低可防止主导 code；提高可容忍市场确实由少数模式主导。",
    ),
    "model.codebook.health.usage_regularization_weight": _config_doc(
        "设置鼓励 code 使用均衡的辅助正则权重（KL-uniform）。",
        "增大可缓解 collapse 但改变 loss；严格复现论文公式时应设为 0。",
    ),
    "model.codebook.health.usage_profit_alignment_weight": _config_doc(
        "设置 code 使用率与轨迹收益正相关的辅助损失权重。",
        "增大可鼓励高收益 archetype 被更多使用；过大可能压制重构目标，严格复现论文公式时应设为 0。",
    ),
    "model.codebook.health.usage_profit_alignment_target_corr": _config_doc(
        "设置 usage-profit alignment 期望达到的最小相关系数。",
        "提高会更强地惩罚收益与使用率负相关；降低更保守，适合先做稳定性验证。",
    ),
    "model.codebook.health.usage_profit_alignment_temperature": _config_doc(
        "设置基于 z_e 与 codebook 距离计算 soft assignment 的 softmax 温度。",
        "增大使软分配更平滑、梯度覆盖更多 code；减小更接近 hard assignment 但信号更尖锐。",
    ),
    "model.codebook.health.dead_code_patience": _config_doc(
        "定义 code 连续未使用多少 epoch 后视为 dead code。",
        "减小会更快重启但可能误判暂时低频 code；增大更保守但恢复慢。",
    ),
    "model.codebook.health.dead_code_restart": _config_doc(
        "控制是否自动重启长期不用的 code。",
        "true 提升可用 code 数；严格复现或诊断原始 collapse 行为时可关闭。",
    ),
    "model.codebook.health.restart_source": _config_doc(
        "定义重启 dead code 时新 embedding 的来源。",
        "高重构误差样本能补足未表达模式；换来源会影响恢复方向。",
    ),
    "model.codebook.health.restart_cooldown_epochs": _config_doc(
        "重启后冷却多少 epoch 再允许 checkpoint 参与 best 选择。",
        "增大更稳但延迟选择；减小更快但可能选到刚扰动后的不稳定模型。",
    ),
    "model.codebook.health.consecutive_collapse_epoch_limit": _config_doc(
        "连续 collapse 多少 epoch 后停止训练并报错。",
        "降低更早失败暴露问题；提高给恢复机制更多时间但浪费算力。",
    ),
    "model.codebook.health.local_optimum_escape": _config_doc(
        "配置低频 code 扰动等逃离局部最优机制。",
        "默认关闭；开启适合 collapse 诊断，但会改变实验条件。",
    ),
    "model.codebook.health.local_optimum_escape.enabled": _config_doc(
        "控制是否启用局部最优逃逸扰动。",
        "true 可帮助低使用率 code 探索；false 保持训练更可复现。",
    ),
    "model.codebook.health.local_optimum_escape.stagnant_usage_epochs": _config_doc(
        "usage ratio 停滞多少 epoch 后触发逃逸条件。",
        "减小更敏感；增大更保守，减少不必要扰动。",
    ),
    "model.codebook.health.local_optimum_escape.stagnant_perplexity_epochs": _config_doc(
        "perplexity 停滞多少 epoch 后触发逃逸条件。",
        "减小更快响应 collapse；增大避免对正常平台期过度干预。",
    ),
    "model.codebook.health.local_optimum_escape.best_score_no_improve_epochs": _config_doc(
        "best score 长期无提升时触发逃逸条件的 epoch 数。",
        "减小更激进；增大更耐心，适合慢收敛训练。",
    ),
    "model.codebook.health.local_optimum_escape.perturbation_probability_per_epoch": _config_doc(
        "满足条件后每个 epoch 执行扰动的概率。",
        "增大更容易跳出局部最优但不稳定；减小更温和。",
    ),
    "model.codebook.health.local_optimum_escape.perturbation_std_ratio": _config_doc(
        "扰动幅度相对 code 向量尺度的比例。",
        "增大探索更强但可能破坏已学模式；减小更安全但效果弱。",
    ),
    "model.codebook.health.local_optimum_escape.min_epochs_between_perturbations": _config_doc(
        "两次扰动之间的最小间隔，避免连续扰动。",
        "增大更稳定；减小恢复更快但训练轨迹更难复现。",
    ),
    "model.codebook.health.local_optimum_escape.only_perturb_low_usage_codes": _config_doc(
        "限制只扰动低使用率 code，保护已稳定的主导 archetype。",
        "true 更安全；false 探索范围更大但可能破坏有效 code。",
    ),

    # 训练
    "training.batch_size": _config_doc(
        "设置训练 batch 大小，影响梯度估计和显存。",
        "增大更稳定但占显存；减小噪声更大但适合小显存。",
    ),
    "training.lr": _config_doc(
        "设置优化器学习率，控制参数更新步幅。",
        "增大收敛快但可能震荡；减小更稳但训练更慢。",
    ),
    "training.epochs": _config_doc(
        "设置最大训练 epoch 数。",
        "增大给模型更多收敛时间但可能过拟合；减小适合快速验证。",
    ),
    "training.pretrain_epochs": _config_doc(
        "设置 Phase A 跳过 VQ 的重构预训练 epoch 数。",
        "增大可让 encoder 初始化 codebook 前学到更稳定表示，但会增加训练时间；严格复现论文公式时自动设为 0。",
    ),
    "training.pretrain_lr": _config_doc(
        "可选设置 Phase A 预训练学习率；为空时复用 training.lr。",
        "单独降低可让预训练更稳；为空保持训练阶段学习率一致，减少调参维度。",
    ),
    "training.seed": _config_doc(
        "固定采样、初始化和训练随机性，保证实验可复现。",
        "改变 seed 可做稳定性检查；对比实验应保持一致。",
    ),
    "training.device": _config_doc(
        "指定训练设备。",
        "cuda 更快但依赖 GPU；cpu 适合调试但训练慢。",
    ),
    "training.save_every": _config_doc(
        "控制 checkpoint 保存间隔。",
        "减小保存更密但产物更多；增大节省空间但可能错过中间好模型。",
    ),
    "training.early_stopping_patience": _config_doc(
        "设置验证指标无提升时提前停止的耐心 epoch 数。",
        "减小更省时但可能早停；增大更完整但更耗时，None 表示不早停。",
    ),
    "training.gradient_clip_norm": _config_doc(
        "限制梯度范数，避免 LSTM/VQ 训练中梯度爆炸。",
        "降低更稳但可能压制学习；提高更自由但数值风险增加。",
    ),
    "training.mixed_precision": _config_doc(
        "控制是否使用混合精度加速训练。",
        "true 更快更省显存；false 数值更直接，适合排查精度问题。",
    ),
    "training.paper_strict_reproduction": _config_doc(
        "标记是否严格复现论文公式，自动关闭部分工程稳定增强。",
        "true 提升论文可比性但稳定性下降；false 使用默认工程防护。",
    ),
    "training.full_validation_every_epochs": _config_doc(
        "控制完整 validation 的执行频率。",
        "减小选择更及时但更慢；增大训练更快但可能延迟发现退化。",
    ),
    "training.fast_val_probe_size": _config_doc(
        "设置快速 validation probe 使用的样本数。",
        "增大 probe 更稳定但更慢；减小更快但指标噪声更大。",
    ),

    # 选择策略
    "selection_policy.selection_metric": _config_doc(
        "指定用于选择 best checkpoint 的主指标。",
        "切换指标会改变提升目标；必须与实验目标和报告解释一致。",
    ),
    "selection_policy.selection_mode": _config_doc(
        "指定主指标是越大越好还是越小越好。",
        "max 用于收益/综合分；min 用于损失类指标，配置错误会选错模型。",
    ),
    "selection_policy.min_code_usage_ratio": _config_doc(
        "best checkpoint 必须满足的 code 使用率下限。",
        "提高更重视 archetype 多样性；降低可接受更集中但风险更高。",
    ),
    "selection_policy.metric_weights": _config_doc(
        "定义综合分中各验证指标的权重。",
        "提高某项会让选择偏向该目标；调整后不同 batch 的综合分不可直接比较。",
    ),
    "selection_policy.risk": _config_doc(
        "配置收益风险类 guardrail。",
        "阈值越严越保守；越松更容易选中高收益但高风险的模型。",
    ),
    "selection_policy.risk.max_drawdown": _config_doc(
        "限制 validation 最大回撤。",
        "降低更保守；提高可容忍更大亏损波动。",
    ),
    "selection_policy.risk.min_sharpe_ratio": _config_doc(
        "设置 validation Sharpe 下限。",
        "提高要求更稳定收益；降低允许收益波动更大的模型。",
    ),
    "selection_policy.behavior": _config_doc(
        "配置行为多样性和 decoder 对 code 敏感性的 guardrail。",
        "阈值越严越能避免无效 archetype；过严可能拒绝真实相近策略。",
    ),
    "selection_policy.behavior.min_inter_code_action_diversity": _config_doc(
        "要求不同 code 解码出的 action 序列有足够差异。",
        "提高会强制 archetype 分化；降低可接受相似但稳定的 code。",
    ),
    "selection_policy.behavior.min_decoder_sensitivity_to_code": _config_doc(
        "要求 decoder 输出对 code 变化有最小敏感度。",
        "提高可防止 decoder 忽略 code；过高可能误伤相近 archetype。",
    ),
    "selection_policy.behavior.min_epoch_code_stability": _config_doc(
        "要求相邻评估中 code assignment 保持一定稳定性。",
        "提高更重视可复现标签；降低允许训练后期仍有迁移。",
    ),
    "selection_policy.teacher": _config_doc(
        "配置 DP teacher 质量 guardrail。",
        "阈值越严越能避免无收益 teacher 训练模型；越松适合低机会市场诊断。",
    ),
    "selection_policy.teacher.min_dp_teacher_profitable_ratio": _config_doc(
        "要求 DP teacher 盈利 horizon 的最小比例。",
        "提高可过滤弱 teacher 批次；降低可保留更困难或低波动市场。",
    ),
    "selection_policy.composite_score_sensitivity_perturbations": _config_doc(
        "配置综合分权重扰动，用于检查 checkpoint 选择是否过度依赖单一权重。",
        "增加扰动可提升稳健性审计；减少扰动更快但诊断较弱。",
    ),

    # 诊断
    "diagnostics.tensorboard_enabled": _config_doc(
        "控制是否写 TensorBoard 日志。",
        "true 便于观察训练曲线；false 减少 IO 和产物体积。",
    ),
    "diagnostics.tensorboard_log_every_epochs": _config_doc(
        "设置 TensorBoard 日志写入频率。",
        "减小更细粒度但 IO 更多；增大更轻量。",
    ),
    "diagnostics.tensorboard_max_points_per_split": _config_doc(
        "限制每个 split 写入可视化的点数。",
        "增大图更完整但文件更大；减小更快但采样代表性下降。",
    ),
    "diagnostics.fixed_probe_seed": _config_doc(
        "固定诊断 probe 抽样 seed，便于跨 epoch 对比同一批样本。",
        "改变会换一组诊断样本；正式对比应保持一致。",
    ),
    "diagnostics.latent_visualization_enabled": _config_doc(
        "控制是否输出 latent 降维可视化。",
        "true 便于检查 code 分离；false 节省时间，尤其是 t-SNE 较慢时。",
    ),
    "diagnostics.latent_projections": _config_doc(
        "指定 latent 可视化使用的降维方法。",
        "pca 快且稳定；tsne 更直观但慢且随机性更强。",
    ),
    "diagnostics.failure_cases_enabled": _config_doc(
        "控制是否输出验证失败样本和高误差案例。",
        "true 便于定位模型弱点；false 减少报告生成成本。",
    ),
    "diagnostics.failure_cases_top_k": _config_doc(
        "设置每类失败案例保留数量。",
        "增大便于人工分析但产物更大；减小更精简。",
    ),
}


def phase1_config_field_docs() -> Dict[str, Dict[str, str]]:
    """返回 Phase I 配置字段说明副本。

    key 使用 ``phase1_config.yaml`` 的点分路径；value 包含:
    - ``why``: 为什么需要这个配置。
    - ``tuning_effect``: 调节这个配置通常会产生什么影响。
    """
    return {path: dict(doc) for path, doc in PHASE1_CONFIG_FIELD_DOCS.items()}


def apply_paper_strict_overrides(config: Phase1Config) -> Phase1Config:
    """``paper_strict_reproduction=True`` 时关闭工程稳定项。

    返回一份新的 Phase1Config。原 config 保持不变（frozen=True）。
    """
    if not config.training.paper_strict_reproduction:
        return config
    new_codebook = replace(
        config.model.codebook,
        init_method="random_normal",
        update_method="gradient",
        health=replace(
            config.model.codebook.health,
            usage_regularization_weight=0.0,
            usage_profit_alignment_weight=0.0,
            dead_code_restart=False,
        ),
    )
    new_encoder_input = replace(
        config.model.encoder_input,
        reward_normalization="train_reward_standard",
        reward_clip_value=5.0,
    )
    new_model = replace(
        config.model,
        beta0=0.25,
        codebook=new_codebook,
        encoder_input=new_encoder_input,
    )
    new_training = replace(config.training, pretrain_epochs=0)
    return replace(config, model=new_model, training=new_training)
