"""Phase I 训练入口。

设计文档锚点: §4.1 与 §10 集成入口。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

# 让 ``python scripts/train_phase1.py`` 可直接运行。
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.phase1_config import (  # noqa: E402
    BehaviorGuardrailConfig,
    CodebookConfig,
    CodebookHealthConfig,
    CostConfig,
    DPConfig,
    DataAugmentationConfig,
    DiagnosticsConfig,
    EncoderInputConfig,
    ModelConfig,
    NoTradeCodeHealthConfig,
    NoTradeControlConfig,
    Phase1Config,
    RiskGuardrailConfig,
    SamplingHealthConfig,
    SelectionPolicyConfig,
    StratificationConfig,
    TrainingConfig,
    apply_paper_strict_overrides,
)
from src.trainers.phase1_trainer import Phase1FatalError, Phase1Trainer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ArchetypeTrader Phase I 训练入口")
    # 路径
    p.add_argument("--pair", required=True)
    p.add_argument("--train-batch-id", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--val-file", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--artifact-root", default="artifacts")
    p.add_argument("--factor-profile", default="short")
    p.add_argument("--factor-list-file", default=None)
    # horizon / 采样
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
    # reward 对齐 / 模型
    p.add_argument(
        "--reward-alignment",
        choices=["paper_formula", "next_row_execution"],
        default="paper_formula",
    )
    p.add_argument("--num-archetypes", type=int, default=10)
    p.add_argument("--code-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--max-position", type=int, default=1)
    # 训练
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--local-smoke-relaxed-guardrails",
        action="store_true",
        help="仅用于本地小样本 smoke: 放宽采样/selection guardrail，生产实验不要使用。",
    )
    p.add_argument(
        "--paper-strict-reproduction",
        action="store_true",
        help="开启时关闭工程稳定项（usage reg / dead-code restart / robust normalization）严格复现论文公式 (4)",
    )
    return p


def build_config(args: argparse.Namespace) -> Phase1Config:
    """把 CLI args 翻译为 ``Phase1Config``。

    实现要点
    --------
    - 不做模型行为默认值偷改; 工程默认全部由 ``Phase1Config`` 字段管。
    - ``paper_strict_reproduction=True`` 时通过 ``apply_paper_strict_overrides``
      自动关闭 usage_regularization / dead_code_restart / robust normalization
      / kmeans_warmup + ema，使训练严格对齐论文公式 (4)。
    - 风险声明字段（``risk_acknowledged_by`` 等）原样塞入 config，
      trainer 与最终报告会在 ``sampling_leakage_diagnostics.json`` 中复述。
    """
    cost = CostConfig(reward_alignment=args.reward_alignment)
    dp = DPConfig(horizon=args.horizon, cost_config=cost, max_position=args.max_position)
    encoder_input = EncoderInputConfig()
    if args.local_smoke_relaxed_guardrails:
        codebook_health = CodebookHealthConfig(
            usage_regularization_weight=0.0,
            dead_code_restart=False,
            consecutive_collapse_epoch_limit=999,
        )
    else:
        codebook_health = CodebookHealthConfig()
    codebook = CodebookConfig(health=codebook_health)
    model = ModelConfig(
        hidden_dim=args.hidden_dim,
        code_dim=args.code_dim,
        num_codes=args.num_archetypes,
        encoder_input=encoder_input,
        codebook=codebook,
    )
    default_training = TrainingConfig()
    training = TrainingConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        paper_strict_reproduction=args.paper_strict_reproduction,
        save_every=(
            1 if args.local_smoke_relaxed_guardrails else default_training.save_every
        ),
        full_validation_every_epochs=(
            1
            if args.local_smoke_relaxed_guardrails
            else default_training.full_validation_every_epochs
        ),
        fast_val_probe_size=(
            8
            if args.local_smoke_relaxed_guardrails
            else default_training.fast_val_probe_size
        ),
    )
    strat = StratificationConfig(
        mode=args.stratification_mode,
        prospective_lookback_minutes=args.prospective_lookback_minutes,
        diagnostic_pair_batch_id=args.diagnostic_pair_batch_id,
    )
    sampling_health = SamplingHealthConfig()
    sampling_overrides = {}
    if args.sampling_min_gap_between_samples is not None:
        sampling_overrides[
            "min_gap_between_samples"
        ] = args.sampling_min_gap_between_samples
    if args.sampling_max_overlap_ratio is not None:
        sampling_overrides["max_overlap_ratio"] = args.sampling_max_overlap_ratio
    if args.sampling_flat_low_vol_max_ratio is not None:
        sampling_overrides["flat_low_vol_max_ratio"] = args.sampling_flat_low_vol_max_ratio
    if args.split_boundary_embargo is not None:
        sampling_overrides["split_boundary_embargo"] = args.split_boundary_embargo
    if args.next_row_split_boundary_embargo is not None:
        sampling_overrides[
            "next_row_split_boundary_embargo"
        ] = args.next_row_split_boundary_embargo
    if args.sampling_health_warn_only:
        sampling_overrides["warn_only"] = True
    if args.sampling_allow_overlap_relaxation:
        sampling_overrides["allow_overlap_relaxation"] = True
    if sampling_overrides:
        sampling_health = replace(sampling_health, **sampling_overrides)
    selection_policy = SelectionPolicyConfig()
    diagnostics = DiagnosticsConfig()
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
        selection_policy = SelectionPolicyConfig(
            min_code_usage_ratio=0.0,
            risk=RiskGuardrailConfig(max_drawdown=10.0, min_sharpe_ratio=-999.0),
            behavior=BehaviorGuardrailConfig(
                min_inter_code_action_diversity=0.0,
                min_decoder_sensitivity_to_code=0.0,
                min_epoch_code_stability=0.0,
            ),
        )
        diagnostics = DiagnosticsConfig(
            failure_cases_enabled=False,
            latent_visualization_enabled=False,
        )

    config = Phase1Config(
        pair=args.pair,
        train_batch_id=args.train_batch_id,
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
        no_trade_control=NoTradeControlConfig(),
        no_trade_code_health=NoTradeCodeHealthConfig(),
        data_augmentation=DataAugmentationConfig(),
        dp=dp,
        model=model,
        training=training,
        selection_policy=selection_policy,
        diagnostics=diagnostics,
        allow_missing_prospective_diagnostic=args.allow_missing_prospective_diagnostic,
        risk_acknowledged_by=args.risk_acknowledged_by,
        expected_sign_off_followup_batch_id=args.expected_sign_off_followup_batch_id,
        local_smoke_relaxed_guardrails=args.local_smoke_relaxed_guardrails,
    )
    return apply_paper_strict_overrides(config)


def assert_prospective_diagnostic(args: argparse.Namespace) -> None:
    """主实验缺 prospective 对照诊断时的第一道防线（CLI 层）。

    Logic
    -----
    - ``stratification_mode=prospective_past``: 这是诊断批次本身，直接放行。
    - ``--diagnostic-pair-batch-id`` 已提供: 放行；trainer 层会再校验一次。
    - 缺 ``--diagnostic-pair-batch-id``:
      * 未传 ``--allow-missing-prospective-diagnostic`` → ``sys.exit(2)`` 并输出指引。
      * 传了 ``--allow-missing-prospective-diagnostic`` 但缺 ``--risk-acknowledged-by``
        或 ``--expected-sign-off-followup-batch-id`` → 同样退出。
    """
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
    config = build_config(args)
    trainer = Phase1Trainer(config)
    try:
        trainer.run()
    except Phase1FatalError as exc:
        logger = getattr(trainer, "_logger", None)
        if logger is not None:
            logger.exception("phase1_fatal_error 说明=Phase I 触发阻塞错误 error=%s", exc)
        print(f"[fatal] Phase I 训练终止: {exc}", file=sys.stderr)
        return 1
    except Exception:
        logger = getattr(trainer, "_logger", None)
        if logger is not None:
            logger.exception("phase1_unexpected_error 说明=Phase I 发生未预期错误")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
