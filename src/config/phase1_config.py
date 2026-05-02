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
    min_gap_between_samples: int = 36  # h=72 时的 50% overlap 上限
    max_overlap_ratio: float = 0.5
    split_boundary_embargo: int = 73            # paper_formula 默认 (h + 1)
    next_row_split_boundary_embargo: int = 74   # next_row_execution 默认 (h + 2)
    warn_only: bool = False
    allow_overlap_relaxation: bool = False


@dataclass(frozen=True)
class NoTradeControlConfig:
    """No-trade 样本处理。"""
    keep_no_trade: bool = True
    max_no_trade_ratio: float = 0.25
    min_profit_gate: float = 0.0
    cap_flat_low_vol_strata: bool = True
    flat_low_vol_max_ratio: float = 0.15
    resample_when_exceeded: bool = True


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
    commission_rate: float = 0.0002  # 论文 δ = 0.02%
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
    ] = "train_reward_robust"
    reward_clip_value: float = 8.0
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
    min_code_usage_ratio: float = 0.7
    max_dominant_code_ratio: float = 0.5
    usage_regularization_weight: float = 0.01
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
    kmeans_warmup_batches: int = 32
    update_method: Literal["gradient", "ema"] = "ema"
    ema_decay: float = 0.99
    ema_epsilon: float = 1.0e-5
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
class Phase1Config:
    """Phase I 顶层配置。"""
    pair: str
    train_batch_id: str
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
    no_trade_control: NoTradeControlConfig = field(default_factory=NoTradeControlConfig)
    no_trade_code_health: NoTradeCodeHealthConfig = field(
        default_factory=NoTradeCodeHealthConfig
    )
    data_augmentation: DataAugmentationConfig = field(
        default_factory=DataAugmentationConfig
    )
    dp: DPConfig = field(default_factory=DPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    selection_policy: SelectionPolicyConfig = field(
        default_factory=SelectionPolicyConfig
    )
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
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
            "sampling_health": SamplingHealthConfig,
            "no_trade_control": NoTradeControlConfig,
            "no_trade_code_health": NoTradeCodeHealthConfig,
            "data_augmentation": DataAugmentationConfig,
            "dp": DPConfig,
            "model": ModelConfig,
            "training": TrainingConfig,
            "selection_policy": SelectionPolicyConfig,
            "diagnostics": DiagnosticsConfig,
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
        codebook=new_codebook,
        encoder_input=new_encoder_input,
    )
    return replace(config, model=new_model)
