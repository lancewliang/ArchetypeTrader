"""Phase I offline preprocessing configuration.

This module owns every configuration field that can change sampled horizons,
DP teacher outputs, processed-data hashes, or the data-process manifest.
Phase I model/optimizer/checkpoint configuration remains in
``src.config.phase1_config``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional


@dataclass(frozen=True)
class StratificationConfig:
    """Stratified sampling configuration."""

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
    """Sampling health thresholds."""

    max_no_trade_ratio: float = 0.25
    flat_low_vol_max_ratio: float = 0.15
    min_gap_between_samples: int = 12
    max_overlap_ratio: float = 0.5
    split_boundary_embargo: int = 73
    next_row_split_boundary_embargo: int = 74
    warn_only: bool = False
    allow_overlap_relaxation: bool = False


@dataclass(frozen=True)
class NoTradeControlConfig:
    """No-trade sample coverage control."""

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
    """Full-time train split sampling configuration."""

    enabled: bool = True
    full_time_mode: Literal["non_overlap", "stride"] = "stride"
    full_time_stride: int = 36
    min_train_ratio: float = 0.40
    label_export_enabled: bool = True


@dataclass(frozen=True)
class EvalLabelingConfig:
    """Validation/test label generation contract."""

    val_mode: Literal["horizon_stride", "all_eligible"] = "horizon_stride"
    test_mode: Literal["horizon_stride", "all_eligible"] = "horizon_stride"
    apply_sampling: bool = False
    apply_augmentation: bool = False


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


@dataclass(frozen=True)
class RejectTransitionHealthConfig:
    """Transition rejection monitoring for insufficient LOB depth."""

    max_horizon_reject_rate: float = 0.10
    max_dataset_reject_rate: float = 0.05
    fail_when_exceeded: bool = True


@dataclass(frozen=True)
class CostConfig:
    """Shared cost configuration for DP teacher and student replay."""

    reward_alignment: Literal["paper_formula", "next_row_execution"] = "paper_formula"
    commission_rate: float = 0.0005
    slippage_model: Literal["lob_depth"] = "lob_depth"
    book_levels: int = 5
    mark_price: Literal["mid_price"] = "mid_price"
    execution_lag: int = 0
    insufficient_depth_policy: Literal["reject_transition"] = "reject_transition"
    reject_transition_health: RejectTransitionHealthConfig = field(
        default_factory=RejectTransitionHealthConfig
    )


@dataclass(frozen=True)
class DPConfig:
    """Single-trade DP teacher configuration."""

    horizon: int = 72
    gamma: float = 1.0
    max_position: int = 1
    cost_config: CostConfig = field(default_factory=CostConfig)


@dataclass(frozen=True)
class Phase1DataProcessConfig:
    """Offline Phase I data processing config."""

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
    dp_workers: int = 0
    dp_worker_chunksize: int = 32
    seed: int = 42
    allow_missing_prospective_diagnostic: bool = False
    risk_acknowledged_by: Optional[str] = None
    expected_sign_off_followup_batch_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def artifacts_dir(self) -> Path:
        return Path(self.artifact_root) / self.pair / self.data_batch_id / "phase1"

