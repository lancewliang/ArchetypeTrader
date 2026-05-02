"""Phase I 训练入口。

训练必须通过 ``--data-process-manifest`` 指向离线数据预处理产物。
数据预处理由 ``scripts/phase1_data_processor.py`` 独立完成。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.phase1_config import (  # noqa: E402
    BehaviorGuardrailConfig,
    CodebookConfig,
    CodebookHealthConfig,
    CostConfig,
    DPConfig,
    DiagnosticsConfig,
    EncoderInputConfig,
    ModelConfig,
    Phase1Config,
    RiskGuardrailConfig,
    SelectionPolicyConfig,
    StratificationConfig,
    TrainingConfig,
    apply_paper_strict_overrides,
)
from src.trainers.phase1_trainer import Phase1FatalError, Phase1Trainer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ArchetypeTrader Phase I 训练入口")
    p.add_argument("--pair", required=True)
    p.add_argument("--train-batch-id", required=True)
    p.add_argument("--data-process-manifest", required=True)
    p.add_argument("--artifact-root", default="artifacts")
    p.add_argument("--factor-profile", default="short")
    p.add_argument("--factor-list-file", default=None)
    p.add_argument("--horizon", type=int, default=72)
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
    p.add_argument(
        "--reward-alignment",
        choices=["paper_formula", "next_row_execution"],
        default="paper_formula",
    )
    p.add_argument("--num-archetypes", type=int, default=10)
    p.add_argument("--code-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--max-position", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--local-smoke-relaxed-guardrails",
        action="store_true",
        help="仅用于本地小样本 smoke: 放宽 selection guardrail，生产实验不要使用。",
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
    selection_policy = SelectionPolicyConfig()
    diagnostics = DiagnosticsConfig()
    if args.local_smoke_relaxed_guardrails:
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
        data_process_manifest=args.data_process_manifest,
        artifact_root=args.artifact_root,
        factor_profile=args.factor_profile,
        factor_list_file=args.factor_list_file,
        horizon=args.horizon,
        stratification=strat,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
