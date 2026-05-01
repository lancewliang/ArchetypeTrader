"""Phase II 集中配置。

设计文档锚点: Phase II 执行计划 §Step 1。

设计原则:
- 镜像 Phase I ``Phase1Config`` 的 frozen dataclass + 显式嵌套类型映射风格。
- 所有默认值必须与 Phase II 设计文档一致。
- 配置必须可序列化为 yaml/json（写入 ``phase2_config.yaml``）。
- ``config_hash`` 必须稳定: 相同字段 → 相同 hash。
- Phase II 不重新训练 Phase I encoder/decoder/codebook，不在线调用 DP。
- Phase II 的 horizon-level reward、成本、换仓和盘口成交语义必须完全复用 ``src/trading/``。
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


# ---------- Phase I 产物引用 ----------

@dataclass(frozen=True)
class Phase1ArtifactsConfig:
    """Phase I 冻结产物路径与校验配置。

    校验项:
    - decoder.pt / codebook.pt / input_schema.json / phase1_report.json 等齐全性。
    - fatal_collapse=false / code_assignment_drift_warning=false。
    - hindsight_bias_warning 检查。
    - cost_config / reward_alignment / max_position 与 Phase I 一致。
    """
    artifact_root: str = "artifacts"
    pair: str = ""
    phase1_batch_id: str = ""
    required_files: List[str] = field(default_factory=lambda: [
        "decoder.pt",
        "codebook.pt",
        "input_schema.json",
        "phase1_report.json",
        "phase1_config.yaml",
        "feature_provenance.json",
        "checkpoint_manifest.json",
    ])


# ---------- Horizon 调度 ----------

@dataclass(frozen=True)
class HorizonScheduleConfig:
    """Horizon index 生成配置。

    支持 non_overlap / stride / phase1_index 三种模式。
    末尾 markout 越界 horizon 必须裁掉。
    gap horizon 必须标注并按配置裁掉。
    """
    mode: Literal["non_overlap", "stride", "phase1_index"] = "non_overlap"
    stride: int = 1
    walk_forward_enabled: bool = True
    walk_forward_seed: int = 42
    gap_threshold_bars: int = 5
    data_gap_check_enabled: bool = True
    max_allowed_gap_minutes: int = 5
    drop_gap_horizons: bool = True
    gap_position_carry_threshold_minutes: int = 5
    gap_large_reset_mode: Literal["carry", "force_flatten", "warmup_only"] = "force_flatten"
    gap_mode: Literal["carry", "force_flatten", "warmup_only"] = "carry"
    exclude_gap_horizons: bool = True
    position_continuity: bool = True
    chunk_reset_position: Literal["inherit", "flat"] = "flat"
    reward_alignment_lookahead_check: bool = True


# ---------- Selector 网络 ----------

@dataclass(frozen=True)
class SelectorNetworkConfig:
    """Archetype selector 网络超参。

    输入 s^sel，输出 K 类离散 logits 与 critic value。
    默认 LayerNorm；RunningMeanStd 只作为 ablation。
    """
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    use_layer_norm: bool = True
    position_continuity: bool = True
    dead_code_mask_source: str = "phase1_global_usage"
    action_mask_dead_codes: bool = True
    dead_code_usage_threshold: float = 0.01
    input_norm: Literal["layer_norm", "running_mean_std", "none"] = "layer_norm"
    position_encoding: Literal["one_hot_3", "scaled_integer", "bucketed_position"] = "scaled_integer"


# ---------- PPO ----------

@dataclass(frozen=True)
class PPOConfig:
    """PPO 训练超参。

    自研实现，不依赖 Stable-Baselines3。
    """
    clip_ratio: float = 0.2
    value_clip_range: Optional[float] = None
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_min_coef: float = 1e-4
    entropy_warmup_coef: Optional[float] = None
    entropy_warmup_fraction: float = 0.0
    kl_demo_coef: float = 1.0  # 论文 α=1
    kl_demo_label_smoothing: float = 0.0
    kl_demo_anneal_to: Optional[float] = None
    kl_demo_anneal_fraction: float = 1.0
    target_kl: Optional[float] = 0.05
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    batch_size: Optional[int] = None
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    advantage_normalization: bool = True
    reward_normalization: bool = False
    lr_schedule: Literal["constant", "linear"] = "linear"
    lr: float = 3e-4


# ---------- Selection Policy ----------

@dataclass(frozen=True)
class Phase2SelectionPolicyConfig:
    """Best selector checkpoint 选择策略。

    guardrails 包含 max_drawdown / min_sharpe / max_turnover_ratio 等。
    val_kl_to_demo / phase1_demo_label_selector_val_net_return 仅作 diagnostic。
    """
    selection_metric: str = "phase2_composite_score"
    primary_metric: str = "val_net_return"
    primary_mode: Literal["max", "min"] = "max"
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "net_return": 1.0,
        "sharpe_ratio": 0.5,
        "max_drawdown": -0.5,
        "turnover": -0.1,
        "action_dominance_ratio": -0.2,
        "active_archetype_ratio": 0.2,
    })
    composite_score_sensitivity_perturbations: List[float] = field(
        default_factory=lambda: [-0.2, 0.2]
    )
    max_drawdown: float = 0.3
    min_sharpe: float = 0.0
    max_turnover_ratio: float = 5.0
    max_action_dominance_ratio: float = 0.8
    min_active_archetype_ratio: float = 0.3
    max_fold_volatility: Optional[float] = None
    min_rolling_worst_fold_score: Optional[float] = None
    require_rolling_result_for_promotion: bool = False


# ---------- Reward Scaling ----------

@dataclass(frozen=True)
class RewardScalingConfig:
    """Reward scaling 配置。

    默认 divide_by_horizon；clip_range=null 表示不裁剪。
    若启用 clip，必须同时记录 clipped/unclipped reward 统计。
    """
    method: Literal["divide_by_horizon", "raw"] = "divide_by_horizon"
    clip_range: Optional[float] = None


@dataclass(frozen=True)
class RewardNormalizationConfig:
    """Reward normalization 配置。

    Phase II 默认不启用 normalization；若启用，训练集统计必须冻结后再用于
    val/test，避免评估泄漏。
    """
    enabled: bool = False
    method: Literal["running_mean_std"] = "running_mean_std"
    freeze_after_fit: bool = True


@dataclass(frozen=True)
class CostAlignmentCheckConfig:
    """Phase I / Phase II 成本配置一致性校验。"""
    enabled: bool = True
    fail_on_mismatch: bool = True


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """早停配置。"""
    enabled: bool = False
    patience: int = 10
    min_delta: float = 0.0
    metric: str = "phase2_composite_score"


@dataclass(frozen=True)
class ResumeConfig:
    """Resume 审计配置。"""
    enabled: bool = True
    require_optimizer_state: bool = True
    require_env_state: bool = True


@dataclass(frozen=True)
class DeploymentLadderConfig:
    """部署阶梯审计配置。"""
    shadow_enabled: bool = False
    paper_enabled: bool = False
    canary_enabled: bool = False


@dataclass(frozen=True)
class EnvShardsConfig:
    """多 env 分片配置。"""
    mode: Literal["contiguous", "round_robin", "rollover"] = "contiguous"
    chunk_reset_position: Literal["inherit", "flat"] = "flat"


@dataclass(frozen=True)
class StateDimBreakdownConfig:
    """selector state 维度审计配置。"""
    enabled: bool = True
    include_feature_columns: bool = True


# ---------- Live Risk Controls ----------

@dataclass(frozen=True)
class LiveRiskControlsConfig:
    """实时风控配置。

    mid_horizon_emergency_flatten 触发时立即结算 liquidation action 及其 cost。
    done/truncated/risk_triggered 组合语义必须固定。
    """
    daily_loss_limit: Optional[float] = None
    rolling_drawdown_limit: Optional[float] = None
    consecutive_loss_limit: Optional[int] = None
    flatten_on_trigger: bool = True
    mid_horizon_emergency_flatten: bool = True
    terminate_episode_on_risk_trigger: bool = False


# ---------- Distribution Shift / OOD ----------

@dataclass(frozen=True)
class DistributionShiftConfig:
    """OOD 检测配置。

    至少支持 zscore / PSI / mahalanobis 之一。
    默认只用 market features，不混入账户状态。
    """
    method: Literal["zscore", "psi", "mahalanobis"] = "zscore"
    threshold: float = 3.0
    use_market_features_only: bool = True
    fallback_action: Literal["flat_only", "conservative"] = "flat_only"


# ---------- Execution Stress ----------

@dataclass(frozen=True)
class ExecutionStressConfig:
    """执行压力测试配置。

    commission / slippage 倍率与 execution_lag 偏移。
    """
    commission_multipliers: List[float] = field(default_factory=lambda: [1.0, 1.5])
    slippage_multipliers: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    execution_lag_offsets: List[int] = field(default_factory=lambda: [0, 1, 2])


# ---------- Rolling Validation ----------

@dataclass(frozen=True)
class RollingValidationConfig:
    """Rolling validation 配置。

    必须在执行层落地为可调用 runner、固定 fold 切法、固定产物与固定验收逻辑。
    不是仅保留配置字段。
    """
    enabled: bool = True
    num_folds: int = 3
    fold_seed: int = 42
    worst_fold_quantile: float = 0.25
    max_fold_volatility: float = 0.5
    min_worst_fold_score: Optional[float] = None


# ---------- Online Action Throttle ----------

@dataclass(frozen=True)
class OnlineActionThrottleConfig:
    """在线动作节流配置。

    限制 selector 的切换频率和置信度门槛。
    """
    min_confidence_for_non_flat_action: float = 0.0
    max_archetype_switches_per_n_horizons: int = 100
    switch_window_n: int = 10
    cooldown_after_large_turnover: int = 0
    max_position_change_per_horizon: Optional[int] = None


# ---------- Numerical Safety ----------

@dataclass(frozen=True)
class NumericalSafetyConfig:
    """数值安全配置。

    tensor 非 finite fail-fast / gradient 爆炸 fail-fast / debug snapshot 导出。
    """
    check_finite: bool = True
    max_gradient_norm: float = 100.0
    fail_fast_on_nan: bool = True
    debug_snapshot_dir: str = "debug_snapshots"


# ---------- 顶层配置 ----------

@dataclass(frozen=True)
class Phase2Config:
    """Phase II 顶层配置。

    输出目录固定为 ``artifacts/{PAIR}/{PHASE2_BATCH_ID}/phase2/``。
    """
    pair: str = ""
    phase1_batch_id: str = ""
    phase2_batch_id: str = ""
    train_file: str = ""
    val_file: str = ""
    test_file: str = ""
    artifact_root: str = "artifacts"
    horizon: int = 72
    max_position: int = 1
    total_timesteps: int = 1_000_000
    num_envs: int = 4
    rollout_length: int = 128
    seed: int = 42
    device: str = "cuda"
    fast_eval_max_horizons: Optional[int] = 256
    fast_eval_stride: Optional[int] = None
    checkpoint_every_updates: Optional[int] = None
    allow_phase1_hindsight_warning: bool = False
    paper_strict_reproduction: bool = False
    resume_from: Optional[str] = None

    phase1_artifacts: Phase1ArtifactsConfig = field(
        default_factory=Phase1ArtifactsConfig
    )
    horizon_schedule: HorizonScheduleConfig = field(
        default_factory=HorizonScheduleConfig
    )
    selector_network: SelectorNetworkConfig = field(
        default_factory=SelectorNetworkConfig
    )
    ppo: PPOConfig = field(default_factory=PPOConfig)
    selection_policy: Phase2SelectionPolicyConfig = field(
        default_factory=Phase2SelectionPolicyConfig
    )
    reward_scaling: RewardScalingConfig = field(default_factory=RewardScalingConfig)
    reward_normalization: RewardNormalizationConfig = field(
        default_factory=RewardNormalizationConfig
    )
    cost_alignment_check: CostAlignmentCheckConfig = field(
        default_factory=CostAlignmentCheckConfig
    )
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    deployment_ladder: DeploymentLadderConfig = field(
        default_factory=DeploymentLadderConfig
    )
    env_shards: EnvShardsConfig = field(default_factory=EnvShardsConfig)
    state_dim_breakdown: StateDimBreakdownConfig = field(
        default_factory=StateDimBreakdownConfig
    )
    live_risk_controls: LiveRiskControlsConfig = field(
        default_factory=LiveRiskControlsConfig
    )
    distribution_shift: DistributionShiftConfig = field(
        default_factory=DistributionShiftConfig
    )
    execution_stress: ExecutionStressConfig = field(
        default_factory=ExecutionStressConfig
    )
    rolling_validation: RollingValidationConfig = field(
        default_factory=RollingValidationConfig
    )
    online_action_throttle: OnlineActionThrottleConfig = field(
        default_factory=OnlineActionThrottleConfig
    )
    numerical_safety: NumericalSafetyConfig = field(
        default_factory=NumericalSafetyConfig
    )

    # ---- 序列化与 hash ----

    def to_dict(self) -> dict:
        """转 dict 以便 yaml/json 序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Phase2Config":
        """从 dict 重建配置（镜像 Phase1Config.from_dict）。"""
        data = dict(payload)
        nested_fields = {
            "phase1_artifacts": Phase1ArtifactsConfig,
            "horizon_schedule": HorizonScheduleConfig,
            "selector_network": SelectorNetworkConfig,
            "ppo": PPOConfig,
            "selection_policy": Phase2SelectionPolicyConfig,
            "reward_scaling": RewardScalingConfig,
            "reward_normalization": RewardNormalizationConfig,
            "cost_alignment_check": CostAlignmentCheckConfig,
            "early_stopping": EarlyStoppingConfig,
            "resume": ResumeConfig,
            "deployment_ladder": DeploymentLadderConfig,
            "env_shards": EnvShardsConfig,
            "state_dim_breakdown": StateDimBreakdownConfig,
            "live_risk_controls": LiveRiskControlsConfig,
            "distribution_shift": DistributionShiftConfig,
            "execution_stress": ExecutionStressConfig,
            "rolling_validation": RollingValidationConfig,
            "online_action_throttle": OnlineActionThrottleConfig,
            "numerical_safety": NumericalSafetyConfig,
        }
        for name, klass in nested_fields.items():
            if name in data and isinstance(data[name], dict):
                data[name] = klass(**data[name])
        config = cls(**data)
        if config.paper_strict_reproduction:
            config = config.apply_paper_strict_overrides()
        return config

    def apply_paper_strict_overrides(self) -> "Phase2Config":
        """应用论文严格复现覆盖项，返回新的 config。

        该方法保持 dataclass frozen 语义，不原地修改。覆盖项仅选择会影响
        Phase II 论文主路径的开关，工程审计/安全字段继续保留。
        """
        return replace(
            self,
            ppo=replace(
                self.ppo,
                kl_demo_coef=1.0,
                entropy_warmup_coef=None,
                entropy_warmup_fraction=0.0,
                reward_normalization=False,
                lr_schedule="linear",
            ),
            selector_network=replace(
                self.selector_network,
                use_layer_norm=True,
                input_norm="layer_norm",
                position_encoding="scaled_integer",
            ),
            horizon_schedule=replace(
                self.horizon_schedule,
                walk_forward_enabled=True,
                position_continuity=True,
            ),
        )

    def write_yaml(self, path: Path) -> Path:
        """原子写 ``phase2_config.yaml``（镜像 Phase1Config.write_yaml）。"""
        import yaml

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
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
        """稳定 hash: 相同字段 → 相同 hash。"""
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def phase1_dir(self) -> Path:
        """Phase I 产物目录。"""
        return (
            Path(self.artifact_root)
            / self.pair
            / self.phase1_batch_id
            / "phase1"
        )

    def artifacts_dir(self) -> Path:
        """Phase II 输出目录: ``artifacts/{PAIR}/{PHASE2_BATCH_ID}/phase2/``。"""
        return (
            Path(self.artifact_root)
            / self.pair
            / self.phase2_batch_id
            / "phase2"
        )
