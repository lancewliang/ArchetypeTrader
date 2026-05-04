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
class RolloutCollectionConfig:
    """PPO rollout 采样执行后端配置。"""
    mode: Literal["serial", "thread", "process"] = "serial"
    max_workers: Optional[int] = None
    fail_fast: bool = True
    process_start_method: Literal["spawn", "forkserver"] = "spawn"
    worker_device: str = "cpu"
    worker_startup_timeout_seconds: float = 60.0
    worker_step_timeout_seconds: Optional[float] = None
    restart_failed_workers: bool = False
    shared_dataset_mode: Literal["pickle", "memmap"] = "pickle"


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
    max_gradient_norm: float = 1000.0
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
    phase1_label_source: Literal["default", "full_time"] = "default"
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
    rollout_collection: RolloutCollectionConfig = field(
        default_factory=RolloutCollectionConfig
    )
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
            "rollout_collection": RolloutCollectionConfig,
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


def _config_doc(why: str, tuning_effect: str) -> Dict[str, str]:
    """构造字段说明；用于生成配置文档和 IDE 检索。"""
    return {"why": why, "tuning_effect": tuning_effect}


# ---------- 配置字段说明 ----------

PHASE2_CONFIG_FIELD_DOCS: Dict[str, Dict[str, str]] = {
    "pair": _config_doc(
        "标识 Phase II 训练对应的交易品种，用于定位 Phase I 产物、日志和输出目录。",
        "换品种时必须调整；同一实验批次内不要混用品种，否则报告和 checkpoint 不可追溯。",
    ),
    "phase1_batch_id": _config_doc(
        "指向上游 Phase I 冻结产物批次，Phase II 会加载其中的 decoder、codebook、schema 和报告。",
        "更换后会改变可选择的 archetype 与输入 schema；只在明确切换上游模型时调整。",
    ),
    "phase2_batch_id": _config_doc(
        "标识本次 Phase II 训练批次，用于隔离 selector checkpoint、报告和 replay log。",
        "新实验或调参 ablation 应使用新 ID；复跑同一实验保持不变便于比对配置 hash。",
    ),
    "train_file": _config_doc(
        "训练 split 的 market data feather 路径，是 PPO rollout 和 selector 学习的唯一市场数据来源。",
        "换训练窗口会改变策略学习分布；正式对比时应固定，只在重切数据或扩样时调整。",
    ),
    "val_file": _config_doc(
        "验证 split 的 market data feather 路径，用于 best 选择、滚动验证和报告指标。",
        "不能用 test 替代；调参期间保持固定，避免把验证集变成隐式测试集。",
    ),
    "test_file": _config_doc(
        "保留给独立 backtest 入口的测试数据路径，Phase II 训练入口不会加载 test 数据。",
        "训练时通常留空或仅用于兼容旧 CLI；最终测试应通过 backtest_phase2.py 单独指定。",
    ),
    "artifact_root": _config_doc(
        "配置 Phase I/II 产物根目录，默认按 artifacts/{pair}/{batch}/phase* 组织。",
        "只影响文件查找和输出位置，不改变模型行为；迁移实验目录时同步调整。",
    ),
    "horizon": _config_doc(
        "定义 selector 每次选择 archetype 后覆盖的时间窗长度，应与 Phase I reward/label 语义一致。",
        "增大更偏中长周期、样本更少且训练更慢；减小更灵敏但交易切换和成本更频繁。",
    ),
    "max_position": _config_doc(
        "设置交易环境允许的最大持仓，必须与 Phase I 成本和 reward 对齐。",
        "增大会放大收益、回撤和盘口深度约束；减小更保守，通常应继承 Phase I 设置。",
    ),
    "total_timesteps": _config_doc(
        "PPO 训练总采样步数，决定训练预算；更新次数约为 total_timesteps / (num_envs * rollout_length)。",
        "增大给策略更多收敛机会但更耗时；减小适合 smoke test 或快速调参。",
    ),
    "num_envs": _config_doc(
        "并行 HorizonEnv 数量，每次 rollout 会同时从多个分片收集样本。",
        "增大提升吞吐但占更多内存，并会减少同一 total_timesteps 下的更新次数；通常与 rollout_length 一起调。",
    ),
    "rollout_length": _config_doc(
        "每个环境在一次 PPO update 前连续采样的步数，单次更新样本量为 num_envs * rollout_length。",
        "增大梯度更稳定但反馈更慢、显存更高；减小更新更频繁但方差更大。若训练震荡可增大，若迭代太慢可减小。",
    ),
    "rollout_collection": _config_doc(
        "配置 PPO rollout 采样后端，控制 env.step 是串行、线程还是多进程执行。",
        "serial 最易复现；process 用于 CPU-bound env.step 吞吐；thread 仅保留作诊断后端。",
    ),
    "rollout_collection.mode": _config_doc(
        "选择 rollout 采样模式：serial 串行，thread 线程，process 常驻子进程 actor。",
        "正式对比实验应固定；排查问题时切回 serial，追求采样吞吐时优先试 process。",
    ),
    "rollout_collection.max_workers": _config_doc(
        "并行采样 worker 数量；process 模式第一版要求与实际 env shard 数一致，空值表示每个 env 一个 worker。",
        "增大 num_envs/worker 可提高吞吐但占更多 CPU/RAM；过大可能因 IPC 和调度开销变慢。",
    ),
    "rollout_collection.fail_fast": _config_doc(
        "并行采样任一 env step 异常时是否立即取消剩余任务并失败。",
        "正式训练应保持 true 以避免部分 rollout 静默进入 buffer；false 仅适合调试异常聚合。",
    ),
    "rollout_collection.process_start_method": _config_doc(
        "process 模式下 multiprocessing 的启动方式。",
        "spawn 更安全，避免 CUDA/fork 隐患；forkserver 可在 Linux 上降低部分启动开销但需单独验证。",
    ),
    "rollout_collection.worker_device": _config_doc(
        "process worker 内 Phase I frozen policy 推理设备。",
        "默认 cpu，避免多个 worker 争用训练 GPU；只有明确验证 GPU worker 安全且更快时才调整。",
    ),
    "rollout_collection.worker_startup_timeout_seconds": _config_doc(
        "process worker 启动和首次 reset 的超时时间。",
        "增大可容忍大数据 pickle 或慢启动；减小能更快暴露卡死 worker。",
    ),
    "rollout_collection.worker_step_timeout_seconds": _config_doc(
        "process worker 单次 step 等待超时；None 表示不设每步超时。",
        "设置后可避免训练永久卡住；过小会误杀较慢 horizon。",
    ),
    "rollout_collection.restart_failed_workers": _config_doc(
        "process worker 失败后是否尝试重启；第一版实现保持 false。",
        "true 会增加恢复复杂度，正式启用前必须验证 checkpoint/env state 一致性。",
    ),
    "rollout_collection.shared_dataset_mode": _config_doc(
        "process worker 获取训练数据的方式，pickle 为首版实现，memmap 预留给大数据优化。",
        "pickle 简单但每个 worker 复制内存；memmap 可降内存但需要额外文件生命周期管理。",
    ),
    "seed": _config_doc(
        "固定环境分片、初始化和训练随机性，保证实验可复现。",
        "对比实验应固定；做稳定性评估时可换多个 seed 重复训练。",
    ),
    "device": _config_doc(
        "指定训练设备，如 cuda 或 cpu。",
        "cuda 更快但依赖 GPU；cpu 适合小样本调试，正式训练通常用 cuda。",
    ),
    "fast_eval_max_horizons": _config_doc(
        "限制快速验证最多评估多少个 horizon，用于降低训练中评估成本。",
        "增大指标更稳但更慢；减小更快但 best 选择噪声更高，None 表示不限制。",
    ),
    "fast_eval_stride": _config_doc(
        "快速验证抽样步长，为空时使用评估器默认策略。",
        "增大可更快扫过长验证集但会跳过更多样本；减小更完整但耗时增加。",
    ),
    "checkpoint_every_updates": _config_doc(
        "控制除 last/best 外是否按固定 update 间隔额外保存周期 checkpoint。",
        "减小可保留更多中间状态但占磁盘；None 或 0 只保留 last/best 路径。",
    ),
    "allow_phase1_hindsight_warning": _config_doc(
        "控制 Phase I 报告中存在 hindsight bias 警告时 Phase II 是否允许继续。",
        "正式实验应保持 false；临时诊断设 true 会降低结论可信度，报告需要明确标记。",
    ),
    "paper_strict_reproduction": _config_doc(
        "启用论文严格复现覆盖项，关闭部分工程增强以靠近论文主路径。",
        "true 提升论文可比性但稳定性可能下降；工程训练和生产候选通常保持 false。",
    ),
    "phase1_label_source": _config_doc(
        "选择 Phase II train split 使用默认 sampled labels 还是 Phase I 导出的 full-time train labels。",
        "full_time 可提升连续时间训练的 label 覆盖；若对应文件缺失会 fail-fast，避免静默退回稀疏标签。",
    ),
    "resume_from": _config_doc(
        "指定从某个 Phase II checkpoint 恢复训练的路径。",
        "用于中断续训；路径为空则从头训练。恢复时应确保配置、数据和 Phase I 产物一致。",
    ),

    "phase1_artifacts": _config_doc(
        "集中配置 Phase I 冻结产物校验要求，确保上游文件齐全且签收状态可审计。",
        "通常只调整 required_files；路径定位主要由顶层 artifact_root/pair/phase1_batch_id 决定。",
    ),
    "phase1_artifacts.artifact_root": _config_doc(
        "Phase I 产物根目录的显式引用字段，保留用于产物校验语义。",
        "一般与顶层 artifact_root 保持一致；当前训练路径以顶层配置为准。",
    ),
    "phase1_artifacts.pair": _config_doc(
        "Phase I 产物所属交易品种的显式引用字段，便于审计上游来源。",
        "一般与顶层 pair 保持一致；不一致会降低配置可读性。",
    ),
    "phase1_artifacts.phase1_batch_id": _config_doc(
        "Phase I 产物批次的显式引用字段，便于审计上游来源。",
        "一般与顶层 phase1_batch_id 保持一致；切换上游模型时同步调整。",
    ),
    "phase1_artifacts.required_files": _config_doc(
        "列出 Phase II 启动前必须存在的 Phase I 文件。",
        "增加文件可提高签收严格度；减少文件只适合临时兼容旧产物，可能削弱校验。",
    ),

    "horizon_schedule": _config_doc(
        "控制 Phase II horizon index 的切分方式、gap 处理和持仓连续性。",
        "这是影响样本数量、时间覆盖和线上语义的核心配置组；对比实验应固定。",
    ),
    "horizon_schedule.mode": _config_doc(
        "选择 horizon 切分模式：non_overlap 不重叠，stride 按步长滑窗，phase1_index 复用 Phase I 索引。",
        "non_overlap 最干净但样本少；stride 样本多但相关性更强；phase1_index 便于复用上游标签。",
    ),
    "horizon_schedule.stride": _config_doc(
        "mode=stride 时相邻 horizon 的起点间隔。",
        "减小会增加样本量和重叠；增大减少相关性和训练成本。",
    ),
    "horizon_schedule.walk_forward_enabled": _config_doc(
        "控制评估是否采用 walk-forward 语义，避免验证过程偷看未来。",
        "正式评估应保持 true；关闭只适合局部调试。",
    ),
    "horizon_schedule.walk_forward_seed": _config_doc(
        "固定 walk-forward 相关抽样或折分随机性。",
        "对比实验保持不变；改变 seed 可做验证稳定性检查。",
    ),
    "horizon_schedule.gap_threshold_bars": _config_doc(
        "按 bar 数识别数据缺口的阈值。",
        "降低更容易标记 gap，训练更保守；提高会容忍更多时间不连续。",
    ),
    "horizon_schedule.data_gap_check_enabled": _config_doc(
        "控制是否检查 market data 的时间缺口。",
        "正式训练应保持 true；关闭会掩盖行情断档导致的 reward 和持仓语义风险。",
    ),
    "horizon_schedule.max_allowed_gap_minutes": _config_doc(
        "允许的最大时间缺口分钟数，超过后 horizon 会被标记为 gap。",
        "降低更严格，适合高频数据；提高适合较稀疏 bar，但会放松连续性假设。",
    ),
    "horizon_schedule.drop_gap_horizons": _config_doc(
        "控制生成索引时是否丢弃包含明显 gap 的 horizon。",
        "true 更稳但样本变少；false 可保留更多样本，但需要后续报告明确 gap 风险。",
    ),
    "horizon_schedule.gap_position_carry_threshold_minutes": _config_doc(
        "控制跨 gap 继承持仓的最大时间阈值。",
        "降低会更常重置或平仓，风险更保守；提高会让持仓连续性更强但可能跨越长断档。",
    ),
    "horizon_schedule.gap_large_reset_mode": _config_doc(
        "大 gap 发生时的持仓处理模式。",
        "force_flatten 最保守；carry 更贴近不中断持仓但风险更高；warmup_only 适合只把 gap 当热身段处理。",
    ),
    "horizon_schedule.gap_mode": _config_doc(
        "普通 gap 的持仓处理模式。",
        "carry 保持连续；force_flatten 降低隔夜/断档风险；warmup_only 会减少有效训练片段。",
    ),
    "horizon_schedule.exclude_gap_horizons": _config_doc(
        "控制评估和报告中是否排除 gap horizon。",
        "true 指标更干净但覆盖更少；false 更全面但可能混入数据质量问题。",
    ),
    "horizon_schedule.position_continuity": _config_doc(
        "控制相邻 horizon 之间是否继承最终持仓。",
        "true 更贴近真实连续交易；false 更像每段独立回测，成本和换仓语义不同。",
    ),
    "horizon_schedule.chunk_reset_position": _config_doc(
        "新分片或 chunk 开始时初始持仓如何设置。",
        "flat 更保守且可复现；inherit 更连续但需要确保分片边界真实相邻。",
    ),
    "horizon_schedule.reward_alignment_lookahead_check": _config_doc(
        "检查 reward 对齐是否会在训练切分中引入未来信息。",
        "正式训练应保持 true；关闭只适合定位旧数据或旧配置兼容问题。",
    ),

    "selector_network": _config_doc(
        "配置 Phase II archetype selector 的网络容量、输入归一化和 action mask。",
        "容量和归一化影响收敛稳定性；action mask 影响是否允许选择 Phase I 死代码。",
    ),
    "selector_network.hidden_dims": _config_doc(
        "设置 selector MLP 隐藏层维度。",
        "增大表达力更强但更慢、更易过拟合；减小更稳但可能欠拟合。",
    ),
    "selector_network.activation": _config_doc(
        "设置 selector 隐藏层激活函数。",
        "relu 简单稳定；切换激活函数会改变优化性质，应作为单独 ablation。",
    ),
    "selector_network.use_layer_norm": _config_doc(
        "控制隐藏层是否使用 LayerNorm。",
        "true 通常更稳；false 更贴近裸 MLP，但对输入尺度更敏感。",
    ),
    "selector_network.position_continuity": _config_doc(
        "控制 selector 输入是否考虑持仓连续性相关信息。",
        "true 让选择器感知当前仓位；false 会削弱换仓成本意识，通常不建议。",
    ),
    "selector_network.dead_code_mask_source": _config_doc(
        "定义死代码掩码的来源，目前默认使用 Phase I 全局 code 使用率。",
        "切换来源会改变哪些 archetype 可选；必须确保报告能解释 mask 依据。",
    ),
    "selector_network.action_mask_dead_codes": _config_doc(
        "控制是否禁止 selector 选择 Phase I 中使用率过低的 code。",
        "true 更安全，可避免无效 archetype；false 允许探索但可能选到 Phase I 未学好的 code。",
    ),
    "selector_network.dead_code_usage_threshold": _config_doc(
        "Phase I code 使用率低于该阈值时视为 dead code。",
        "提高会屏蔽更多 code、更保守；降低会放开更多 archetype 供策略选择。",
    ),
    "selector_network.input_norm": _config_doc(
        "设置 selector 输入归一化方式。",
        "layer_norm 默认稳健；none 只适合排查归一化影响；running_mean_std 当前属于消融路径。",
    ),
    "selector_network.position_encoding": _config_doc(
        "设置持仓状态进入 selector 的编码方式。",
        "scaled_integer 简洁；one_hot_3 更离散清晰；bucketed_position 适合未来更大持仓范围。",
    ),

    "ppo": _config_doc(
        "集中配置 PPO 优化、探索熵、KL-demo 约束和学习率调度。",
        "这是最直接影响训练稳定性和收益表现的配置组；一次只调整少数关键参数。",
    ),
    "ppo.clip_ratio": _config_doc(
        "PPO policy ratio 裁剪范围，限制单次策略更新幅度。",
        "降低更保守但学习慢；提高学习更快但更容易策略崩塌。",
    ),
    "ppo.value_clip_range": _config_doc(
        "可选的 value function 裁剪范围。",
        "设置后可稳定 critic 更新；为空表示不裁剪，若 value_loss 震荡可尝试开启。",
    ),
    "ppo.value_loss_coef": _config_doc(
        "critic value loss 在总损失中的权重。",
        "增大更重视价值估计但可能压制策略更新；减小更偏 actor，但优势估计可能变差。",
    ),
    "ppo.entropy_coef": _config_doc(
        "策略熵奖励初始系数，鼓励 selector 探索不同 archetype。",
        "增大探索更多但收益收敛慢；减小更贪心，若 action 过早单一可增大。",
    ),
    "ppo.entropy_min_coef": _config_doc(
        "熵系数退火后的下限。",
        "提高可长期保持探索；降低让后期更确定，可能加剧 action dominance。",
    ),
    "ppo.entropy_warmup_coef": _config_doc(
        "训练早期可选的更高熵系数。",
        "设置后早期探索更强；为空则不启用 warmup，若初始选择过于集中可尝试。",
    ),
    "ppo.entropy_warmup_fraction": _config_doc(
        "熵 warmup 覆盖的训练进度比例。",
        "增大延长探索期；过大可能延迟收益收敛。",
    ),
    "ppo.kl_demo_coef": _config_doc(
        "Phase I demo label KL 约束权重，控制 selector 向 teacher/label 靠拢的强度。",
        "增大更像 Phase I demo、风险更稳但创新少；减小更自由，收益可能更高但偏离 teacher。",
    ),
    "ppo.kl_demo_label_smoothing": _config_doc(
        "对 demo label 做平滑，避免 KL 目标过硬。",
        "增大可缓解噪声标签但弱化 teacher 信号；通常从 0 到小值做消融。",
    ),
    "ppo.kl_demo_anneal_to": _config_doc(
        "KL-demo 系数退火终值；为空时调度器使用默认退火规则。",
        "降低终值让后期更自由；保持较高终值让策略持续贴近 demo。",
    ),
    "ppo.kl_demo_anneal_fraction": _config_doc(
        "KL-demo 系数完成退火所用的训练进度比例。",
        "减小更早放开策略；增大更长时间受 demo 约束。",
    ),
    "ppo.target_kl": _config_doc(
        "PPO 更新中的近似 KL 上限，超过后提前停止当前 update。",
        "降低更保守且训练可能变慢；提高允许更大步更新但不稳定风险增加。",
    ),
    "ppo.max_grad_norm": _config_doc(
        "PPO 反向传播时的梯度范数裁剪阈值。",
        "降低更稳但可能欠更新；提高更自由但要关注数值安全和 loss 爆炸。",
    ),
    "ppo.update_epochs": _config_doc(
        "每次 rollout 后重复优化同一批样本的 epoch 数。",
        "增大样本利用率但更容易过拟合旧 rollout；减小更保守但样本利用不足。",
    ),
    "ppo.batch_size": _config_doc(
        "预留的 PPO 总 batch 大小字段；当前实现主要由 num_envs * rollout_length 决定。",
        "通常保持 None；若未来接入显式 batch 逻辑，再用它覆盖 rollout 样本量。",
    ),
    "ppo.minibatch_size": _config_doc(
        "PPO update 中每个小批次的样本数。",
        "增大梯度更稳定但占显存；减小适合小显存但梯度噪声更大。",
    ),
    "ppo.gamma": _config_doc(
        "折扣因子，控制未来 reward 在 GAE/return 中的权重。",
        "增大更看重长期效果；减小更偏短期，horizon 较短时通常不宜过低。",
    ),
    "ppo.gae_lambda": _config_doc(
        "GAE lambda，控制优势估计的 bias/variance 权衡。",
        "提高方差更大但偏差小；降低更平滑但可能低估长期收益。",
    ),
    "ppo.advantage_normalization": _config_doc(
        "控制是否标准化 advantage。",
        "true 通常更稳；false 只适合排查标准化对收益尺度的影响。",
    ),
    "ppo.reward_normalization": _config_doc(
        "PPO 内 reward normalization 开关；当前 Phase II 尚未实现 running_mean_std。",
        "保持 false；若设 true 会 fail-fast，应使用 reward_scaling 处理 reward 尺度。",
    ),
    "ppo.lr_schedule": _config_doc(
        "学习率调度策略。",
        "linear 后期更稳；constant 保持学习强度，可能更快也更容易震荡。",
    ),
    "ppo.lr": _config_doc(
        "Adam 优化器学习率。",
        "增大收敛快但易震荡；减小更稳但慢。若 approx_kl 频繁超阈值可降低。",
    ),

    "selection_policy": _config_doc(
        "配置 best selector 的选择指标、综合分权重和风险/行为护栏。",
        "调权重会改变最佳 checkpoint；调阈值会改变哪些模型能被签收。",
    ),
    "selection_policy.selection_metric": _config_doc(
        "指定用于 best 选择的最终指标名。",
        "默认使用综合分；切换指标后不同 batch 的 best 选择不可直接比较。",
    ),
    "selection_policy.primary_metric": _config_doc(
        "best history 恢复或兜底时使用的主指标。",
        "通常与收益目标保持一致；如果 selection_metric 缺失会用它辅助判断。",
    ),
    "selection_policy.primary_mode": _config_doc(
        "声明主指标是越大越好还是越小越好。",
        "收益/综合分用 max；损失类指标用 min，配置错误会选反 checkpoint。",
    ),
    "selection_policy.metric_weights": _config_doc(
        "综合分中各验证指标的权重。",
        "提高某项会让选择更偏向该目标；权重改动后要重新做敏感性分析。",
    ),
    "selection_policy.composite_score_sensitivity_perturbations": _config_doc(
        "综合分权重扰动比例，用于检查 best 选择是否过度依赖单一权重。",
        "增加扰动更严格但报告更多；减少扰动更快但稳健性诊断较弱。",
    ),
    "selection_policy.max_drawdown": _config_doc(
        "允许的验证最大回撤上限。",
        "降低更保守；提高可容忍更大波动但风险更高。",
    ),
    "selection_policy.min_sharpe": _config_doc(
        "验证 Sharpe 下限。",
        "提高要求收益更稳定；降低允许噪声更大的高收益候选。",
    ),
    "selection_policy.max_turnover_ratio": _config_doc(
        "允许的换手率上限，防止策略过度频繁切换 archetype/仓位。",
        "降低可控制成本和行为抖动；提高会放开更活跃策略但成本风险更大。",
    ),
    "selection_policy.max_action_dominance_ratio": _config_doc(
        "限制单一 action/archetype 在验证中占比过高。",
        "降低可防止策略塌缩；提高可容忍市场确实偏向少数动作。",
    ),
    "selection_policy.min_active_archetype_ratio": _config_doc(
        "要求验证中活跃 archetype 的最小比例。",
        "提高更重视多样性；降低可接受更集中但可能退化成单一策略。",
    ),
    "selection_policy.max_fold_volatility": _config_doc(
        "滚动验证各 fold 指标波动上限。",
        "降低更要求跨时间稳定；为空表示不额外限制。",
    ),
    "selection_policy.min_rolling_worst_fold_score": _config_doc(
        "滚动验证最差 fold 分数下限。",
        "提高会过滤只在局部时期表现好的模型；为空表示不启用该门槛。",
    ),
    "selection_policy.require_rolling_result_for_promotion": _config_doc(
        "控制提升 best 前是否必须有滚动验证结果。",
        "true 更严格但训练中评估更慢；false 先按快速验证选择，再在报告中诊断。",
    ),

    "reward_scaling": _config_doc(
        "配置训练 reward 尺度变换，避免 PPO loss 被 horizon 累积收益尺度主导。",
        "优先用该配置处理 reward 尺度；不要启用尚未实现的 reward_normalization。",
    ),
    "reward_scaling.method": _config_doc(
        "选择 reward scaling 方法。",
        "divide_by_horizon 让不同 horizon 长度更可比；raw 保留原始收益尺度但训练可能更不稳。",
    ),
    "reward_scaling.clip_range": _config_doc(
        "可选裁剪缩放后 reward 的绝对值范围。",
        "设置后可抑制极端样本；过小会剪掉关键行情信号，None 表示不裁剪。",
    ),
    "reward_normalization": _config_doc(
        "预留的 reward normalization 配置组，当前训练路径尚未实现。",
        "保持 disabled；开启会触发 fail-fast，避免未冻结统计造成泄漏。",
    ),
    "reward_normalization.enabled": _config_doc(
        "控制是否启用 running mean/std reward normalization。",
        "当前必须保持 false；需要归一化时先实现冻结统计和无泄漏验证。",
    ),
    "reward_normalization.method": _config_doc(
        "reward normalization 方法名，当前仅预留 running_mean_std。",
        "暂不建议调整；实现前不会进入训练主路径。",
    ),
    "reward_normalization.freeze_after_fit": _config_doc(
        "要求 normalization 统计在训练集拟合后冻结，再用于 val/test。",
        "实现后应保持 true；false 会带来评估泄漏风险。",
    ),

    "cost_alignment_check": _config_doc(
        "控制 Phase I/II 成本配置一致性检查。",
        "正式训练应开启，确保 selector 学到的行为和 Phase I teacher 成本语义一致。",
    ),
    "cost_alignment_check.enabled": _config_doc(
        "是否执行成本配置一致性校验。",
        "关闭只适合兼容旧产物；正式实验关闭会降低报告可信度。",
    ),
    "cost_alignment_check.fail_on_mismatch": _config_doc(
        "成本不一致时是否直接失败。",
        "true 可防止错误训练；false 仅适合诊断不一致来源。",
    ),

    "early_stopping": _config_doc(
        "配置验证指标长期无提升时提前停止训练。",
        "开启可节省算力但可能错过后期恢复；关闭更完整但更耗时。",
    ),
    "early_stopping.enabled": _config_doc(
        "是否启用早停。",
        "小样本调参可开启；正式长训练可先关闭观察完整曲线。",
    ),
    "early_stopping.patience": _config_doc(
        "验证指标连续多少次评估无改善后停止。",
        "减小更省时但更容易误停；增大更耐心但训练更久。",
    ),
    "early_stopping.min_delta": _config_doc(
        "判定指标改善所需的最小变化量。",
        "增大可忽略噪声小波动；减小更敏感但可能频繁重置 patience。",
    ),
    "early_stopping.metric": _config_doc(
        "用于早停判断的指标名。",
        "应与 selection_metric 或核心收益目标一致；切换后早停行为会改变。",
    ),

    "resume": _config_doc(
        "配置 checkpoint 恢复训练时必须审计的状态。",
        "越严格越能保证续训一致性；放宽只适合旧 checkpoint 兼容。",
    ),
    "resume.enabled": _config_doc(
        "声明是否支持恢复训练路径。",
        "保持 true 便于中断续训；false 通常只用于禁用相关审计显示。",
    ),
    "resume.require_optimizer_state": _config_doc(
        "恢复时是否要求 checkpoint 包含优化器状态。",
        "true 续训更一致；false 可加载旧模型但学习率动量会重置。",
    ),
    "resume.require_env_state": _config_doc(
        "恢复时是否要求 checkpoint 包含环境状态。",
        "true 可减少 rollout 断点偏差；false 适合只想从模型权重继续训练。",
    ),

    "deployment_ladder": _config_doc(
        "记录 shadow/paper/canary 部署阶梯状态，供报告审计。",
        "这些开关不应替代真实部署流程；开启表示该 batch 准备进入对应阶段。",
    ),
    "deployment_ladder.shadow_enabled": _config_doc(
        "标记是否进入 shadow 观察阶段。",
        "true 表示只观察不交易；正式上线前建议先通过 shadow。",
    ),
    "deployment_ladder.paper_enabled": _config_doc(
        "标记是否进入 paper trading 阶段。",
        "true 表示可进行模拟交易审计；需先确认报告护栏通过。",
    ),
    "deployment_ladder.canary_enabled": _config_doc(
        "标记是否进入小流量 canary 阶段。",
        "true 风险最高，应只在 shadow/paper 通过后由人工签收。",
    ),

    "env_shards": _config_doc(
        "控制训练数据如何切分到多个 HorizonEnv。",
        "影响样本顺序、持仓连续性和并行效率；对比实验应固定。",
    ),
    "env_shards.mode": _config_doc(
        "选择分片策略：contiguous 连续切块，round_robin 轮转分配，rollover 预留滚动语义。",
        "contiguous 保持时间局部性；round_robin 更均衡但弱化连续时间结构。",
    ),
    "env_shards.chunk_reset_position": _config_doc(
        "每个 env 分片开始时初始持仓处理方式。",
        "flat 更保守；inherit 只有在分片边界语义明确连续时才使用。",
    ),

    "state_dim_breakdown": _config_doc(
        "配置 selector state 维度审计输出。",
        "开启便于检查输入 schema 和持仓编码；关闭可减少报告细节。",
    ),
    "state_dim_breakdown.enabled": _config_doc(
        "是否输出 state 维度拆解。",
        "正式报告建议保持 true；调试最小输出时可关闭。",
    ),
    "state_dim_breakdown.include_feature_columns": _config_doc(
        "是否在维度审计中包含特征列信息。",
        "true 更易定位 schema 变化；false 可减少报告体积。",
    ),

    "live_risk_controls": _config_doc(
        "配置实时风控阈值和触发后的环境语义。",
        "阈值越严越保守；这些配置会影响训练和回测中的风险触发记录。",
    ),
    "live_risk_controls.daily_loss_limit": _config_doc(
        "单日亏损上限，超过后触发风控。",
        "降低更保守；为空表示不启用该限制。",
    ),
    "live_risk_controls.rolling_drawdown_limit": _config_doc(
        "滚动回撤上限，超过后触发风控。",
        "降低可更快止损；过低可能频繁打断正常波动。",
    ),
    "live_risk_controls.consecutive_loss_limit": _config_doc(
        "连续亏损次数上限。",
        "降低更快进入保护；为空表示不按连续亏损次数限制。",
    ),
    "live_risk_controls.flatten_on_trigger": _config_doc(
        "风控触发时是否立即平仓。",
        "true 更安全且成本会被记录；false 只适合分析触发但不干预的场景。",
    ),
    "live_risk_controls.mid_horizon_emergency_flatten": _config_doc(
        "是否允许 horizon 中途触发紧急平仓。",
        "true 更贴近实盘风控；false 会让风险处理只在边界发生，可能低估损失。",
    ),
    "live_risk_controls.terminate_episode_on_risk_trigger": _config_doc(
        "风控触发后是否结束当前 episode。",
        "true 让风险事件成为 episode 终点；false 保持连续训练但需要正确记录 risk_triggered。",
    ),

    "distribution_shift": _config_doc(
        "配置训练分布到验证/测试分布的漂移检测。",
        "阈值越严越容易发现 OOD；过严会增加报告警告，需要结合样本覆盖解释。",
    ),
    "distribution_shift.method": _config_doc(
        "选择分布漂移检测方法。",
        "zscore 简单稳定；psi/mahalanobis 适合未来更复杂的分布诊断。",
    ),
    "distribution_shift.threshold": _config_doc(
        "分布漂移告警阈值。",
        "降低更敏感；提高减少告警但可能漏掉 OOD。",
    ),
    "distribution_shift.use_market_features_only": _config_doc(
        "检测分布漂移时是否只使用市场特征，排除账户/持仓状态。",
        "true 更聚焦数据分布；false 会把策略行为也混入 OOD 诊断。",
    ),
    "distribution_shift.fallback_action": _config_doc(
        "线上遇到 OOD 时建议的 fallback 动作策略。",
        "flat_only 最保守；conservative 可保留部分交易能力但风险更高。",
    ),

    "execution_stress": _config_doc(
        "配置报告中的执行压力测试场景，如手续费、滑点和成交延迟。",
        "场景越多越能暴露执行风险，但回测耗时和报告体积会增加。",
    ),
    "execution_stress.commission_multipliers": _config_doc(
        "手续费压力测试倍率列表。",
        "增加更高倍率可检查成本敏感性；倍率过高可能只用于极端风险边界。",
    ),
    "execution_stress.slippage_multipliers": _config_doc(
        "滑点压力测试倍率列表。",
        "增大可模拟流动性恶化；列表越长回测越慢。",
    ),
    "execution_stress.execution_lag_offsets": _config_doc(
        "成交延迟偏移列表。",
        "增加偏移可检查信号对延迟的敏感性；过大偏移可能不符合当前频率假设。",
    ),

    "rolling_validation": _config_doc(
        "配置滚动验证折分和提升 best 时的稳定性护栏。",
        "开启可防止只在某段验证期表现好；关闭会降低时间稳健性审计。",
    ),
    "rolling_validation.enabled": _config_doc(
        "是否启用滚动验证。",
        "正式报告建议保持 true；快速 smoke test 可关闭以减少耗时。",
    ),
    "rolling_validation.num_folds": _config_doc(
        "滚动验证折数。",
        "增加折数更细但每折样本更少；减少折数更稳但时间分辨率低。",
    ),
    "rolling_validation.fold_seed": _config_doc(
        "固定滚动验证折分随机性。",
        "对比实验保持固定；改变 seed 可测试折分稳定性。",
    ),
    "rolling_validation.worst_fold_quantile": _config_doc(
        "报告中用于衡量较差 fold 的分位点。",
        "降低更关注尾部风险；提高更接近平均表现。",
    ),
    "rolling_validation.max_fold_volatility": _config_doc(
        "允许各 fold 分数波动的上限。",
        "降低更严格；提高可容忍市场阶段差异。",
    ),
    "rolling_validation.min_worst_fold_score": _config_doc(
        "最差 fold 分数下限。",
        "提高可过滤阶段性失效模型；为空表示不启用该限制。",
    ),

    "online_action_throttle": _config_doc(
        "配置线上动作节流，限制低置信度交易和频繁 archetype 切换。",
        "阈值越严越保守、成本越低；过严可能错过短期机会。",
    ),
    "online_action_throttle.min_confidence_for_non_flat_action": _config_doc(
        "非空仓动作所需的最小置信度。",
        "提高会减少低置信度交易；降低会更积极但噪声交易更多。",
    ),
    "online_action_throttle.max_archetype_switches_per_n_horizons": _config_doc(
        "在指定窗口内允许的最大 archetype 切换次数。",
        "降低可减少行为抖动和成本；提高允许更灵活切换。",
    ),
    "online_action_throttle.switch_window_n": _config_doc(
        "动作切换计数使用的 horizon 窗口大小。",
        "增大更平滑；减小对短期频繁切换更敏感。",
    ),
    "online_action_throttle.cooldown_after_large_turnover": _config_doc(
        "发生大换手后冷却多少个 horizon。",
        "增大可防止连续大幅换仓；0 表示不启用冷却。",
    ),
    "online_action_throttle.max_position_change_per_horizon": _config_doc(
        "单个 horizon 内允许的最大仓位变化。",
        "降低更保守；为空表示不额外限制仓位变化。",
    ),

    "numerical_safety": _config_doc(
        "配置训练中的数值安全检查和 debug snapshot 输出。",
        "越严格越早暴露 NaN/梯度爆炸；关闭只适合定位误报。",
    ),
    "numerical_safety.check_finite": _config_doc(
        "是否检查 tensor 和 loss 为有限值。",
        "正式训练应保持 true；关闭可能让 NaN 延迟到更难定位的阶段。",
    ),
    "numerical_safety.max_gradient_norm": _config_doc(
        "数值安全层面的梯度爆炸阈值，区别于 PPO 更新中的裁剪阈值。",
        "降低更敏感；提高减少误报但可能漏掉异常梯度。默认值应显著高于 ppo.max_grad_norm，因为该检查发生在梯度裁剪前。",
    ),
    "numerical_safety.fail_fast_on_nan": _config_doc(
        "发现 NaN/Inf 时是否立即失败。",
        "true 便于保留第一现场；false 只适合探索性调试。",
    ),
    "numerical_safety.debug_snapshot_dir": _config_doc(
        "数值异常时写入 debug snapshot 的目录。",
        "调整只影响诊断输出位置；正式运行建议放在产物目录或易清理的位置。",
    ),
}


def phase2_config_field_docs() -> Dict[str, Dict[str, str]]:
    """返回 Phase II 配置字段说明副本。

    key 使用 ``phase2_config.yaml`` 的点分路径；value 包含:
    - ``why``: 参数含义、为什么需要这个配置。
    - ``tuning_effect``: 这个参数应该如何调整，以及调整后的常见影响。
    """
    return {path: dict(doc) for path, doc in PHASE2_CONFIG_FIELD_DOCS.items()}
