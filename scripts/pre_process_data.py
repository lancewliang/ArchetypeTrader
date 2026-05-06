"""Phase I offline data preprocessing CLI."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess_data.config import (  # noqa: E402
    CostConfig,
    DPConfig,
    DataAugmentationConfig,
    EvalLabelingConfig,
    NoTradeControlConfig,
    Phase1DataProcessConfig,
    SamplingHealthConfig,
    StratificationConfig,
    TimeDistributionSamplingConfig,
)
from src.preprocess_data.processor import (  # noqa: E402
    Phase1DataProcessor,
    Phase1FatalError,
)
from src.utils.run_logging import configure_run_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """定义 CLI 参数，并把实验契约显式暴露给运行命令。"""
    p = argparse.ArgumentParser(description="ArchetypeTrader Phase I 数据预处理入口")
    p.add_argument("--pair", required=True)
    p.add_argument("--data-batch-id", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--val-file", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--artifact-root", default="artifacts")
    p.add_argument("--factor-profile", default="short")
    p.add_argument("--factor-list-file", default=None)
    p.add_argument("--horizon", type=int, default=72)
    p.add_argument("--num-demos", type=int, default=30000)
    p.add_argument(
        "--sampling-strategy",
        choices=["stratified_uniform", "stratified_proportional"],
        default="stratified_uniform",
    )
    p.add_argument(
        "--stratification-mode",
        choices=["hindsight_horizon", "prospective_past"],
        default="hindsight_horizon",
    )
    p.add_argument("--diagnostic-pair-batch-id", default=None)
    p.add_argument("--allow-missing-prospective-diagnostic", action="store_true")
    p.add_argument("--risk-acknowledged-by", default=None)
    p.add_argument("--expected-sign-off-followup-batch-id", default=None)
    p.add_argument("--prospective-lookback-minutes", type=int, default=1440)
    p.add_argument("--sampling-min-gap-between-samples", type=int, default=None)
    p.add_argument("--sampling-max-overlap-ratio", type=float, default=None)
    p.add_argument("--sampling-flat-low-vol-max-ratio", type=float, default=None)
    p.add_argument("--split-boundary-embargo", type=int, default=None)
    p.add_argument("--next-row-split-boundary-embargo", type=int, default=None)
    p.add_argument("--sampling-health-warn-only", action="store_true")
    p.add_argument("--sampling-allow-overlap-relaxation", action="store_true")
    p.add_argument(
        "--reward-alignment",
        choices=["paper_formula", "next_row_execution"],
        default="paper_formula",
    )
    p.add_argument("--max-position", type=int, default=1)
    p.add_argument(
        "--dp-workers",
        type=int,
        default=0,
        help="DP teacher 并行 worker 数；0=大样本自动使用可用 CPU 核。",
    )
    p.add_argument(
        "--dp-worker-chunksize",
        type=int,
        default=32,
        help="DP 多进程 map chunksize；大样本可适当调大以降低调度开销。",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--local-smoke-relaxed-guardrails",
        action="store_true",
        help="仅用于本地小样本 smoke: 放宽采样 guardrail，生产实验不要使用。",
    )
    p.add_argument(
        "--min-no-trade-ratio",
        type=float,
        default=None,
        help="覆盖 no_trade 样本最小比例要求 (默认 0.10)。",
    )
    p.add_argument(
        "--min-low-opportunity-ratio",
        type=float,
        default=None,
        help="覆盖低机会样本最小比例要求 (默认 0.25)。",
    )
    return p


def build_data_process_config(args: argparse.Namespace) -> Phase1DataProcessConfig:
    """把 CLI 参数转换成强类型配置对象。"""
    cost = CostConfig(reward_alignment=args.reward_alignment)
    dp = DPConfig(horizon=args.horizon, cost_config=cost, max_position=args.max_position)
    strat = StratificationConfig(
        mode=args.stratification_mode,
        prospective_lookback_minutes=args.prospective_lookback_minutes,
        diagnostic_pair_batch_id=args.diagnostic_pair_batch_id,
    )
    sampling_health = SamplingHealthConfig()
    overrides = {}
    if args.sampling_min_gap_between_samples is not None:
        overrides["min_gap_between_samples"] = args.sampling_min_gap_between_samples
    if args.sampling_max_overlap_ratio is not None:
        overrides["max_overlap_ratio"] = args.sampling_max_overlap_ratio
    if args.sampling_flat_low_vol_max_ratio is not None:
        overrides["flat_low_vol_max_ratio"] = args.sampling_flat_low_vol_max_ratio
    if args.split_boundary_embargo is not None:
        overrides["split_boundary_embargo"] = args.split_boundary_embargo
    if args.next_row_split_boundary_embargo is not None:
        overrides["next_row_split_boundary_embargo"] = args.next_row_split_boundary_embargo
    if args.sampling_health_warn_only:
        overrides["warn_only"] = True
    if args.sampling_allow_overlap_relaxation:
        overrides["allow_overlap_relaxation"] = True
    if overrides:
        sampling_health = replace(sampling_health, **overrides)
    if args.local_smoke_relaxed_guardrails:
        sampling_health = SamplingHealthConfig(
            max_no_trade_ratio=1.0,
            flat_low_vol_max_ratio=1.0,
            min_gap_between_samples=1,
            max_overlap_ratio=1.0,
            split_boundary_embargo=0,
            next_row_split_boundary_embargo=0,
            warn_only=True,
        )
    no_trade_control = NoTradeControlConfig()
    nt_overrides = {}
    if args.min_no_trade_ratio is not None:
        nt_overrides["min_no_trade_ratio"] = args.min_no_trade_ratio
    if args.min_low_opportunity_ratio is not None:
        nt_overrides["min_low_opportunity_ratio"] = args.min_low_opportunity_ratio
    if args.local_smoke_relaxed_guardrails:
        nt_overrides.update({
            "max_no_trade_ratio": 1.0,
            "min_no_trade_ratio": 0.0,
            "min_low_opportunity_ratio": 0.0,
            "flat_low_vol_max_ratio": 1.0,
        })
    if nt_overrides:
        no_trade_control = replace(no_trade_control, **nt_overrides)
    return Phase1DataProcessConfig(
        pair=args.pair,
        data_batch_id=args.data_batch_id,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        artifact_root=args.artifact_root,
        factor_profile=args.factor_profile,
        factor_list_file=args.factor_list_file,
        horizon=args.horizon,
        num_demos=args.num_demos,
        sampling_strategy=args.sampling_strategy,
        stratification=strat,
        sampling_health=sampling_health,
        no_trade_control=no_trade_control,
        time_distribution_sampling=TimeDistributionSamplingConfig(),
        eval_labeling=EvalLabelingConfig(),
        data_augmentation=DataAugmentationConfig(),
        dp=dp,
        dp_workers=int(getattr(args, "dp_workers", 0)),
        dp_worker_chunksize=max(1, int(getattr(args, "dp_worker_chunksize", 32))),
        seed=args.seed,
        allow_missing_prospective_diagnostic=args.allow_missing_prospective_diagnostic,
        risk_acknowledged_by=args.risk_acknowledged_by,
        expected_sign_off_followup_batch_id=args.expected_sign_off_followup_batch_id,
    )


def assert_prospective_diagnostic(args: argparse.Namespace) -> None:
    """在 CLI 入口处阻止缺少 prospective 对照的 hindsight 主实验。"""
    if args.stratification_mode == "prospective_past":
        return
    if args.diagnostic_pair_batch_id:
        return
    if args.allow_missing_prospective_diagnostic:
        if not args.risk_acknowledged_by or not args.expected_sign_off_followup_batch_id:
            print(
                "[error] allow_missing_prospective_diagnostic 需配套 "
                "--risk-acknowledged-by + --expected-sign-off-followup-batch-id",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return
    print(
        "[error] hindsight 主实验缺少 --diagnostic-pair-batch-id; "
        "请同时配套一个 prospective_past BATCH_ID 或显式风险声明。",
        file=sys.stderr,
    )
    raise SystemExit(2)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    assert_prospective_diagnostic(args)
    config = build_data_process_config(args)
    configure_run_logger(
        phase="data_process",
        pair=config.pair,
        batch_id=config.data_batch_id,
    )
    processor = Phase1DataProcessor(config)
    try:
        manifest = processor.run()
    except Phase1FatalError as exc:
        print(f"[fatal] Phase I 数据预处理终止: {exc}", file=sys.stderr)
        return 1
    print(f"data_process_manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
