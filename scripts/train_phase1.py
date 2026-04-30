"""Phase I 训练入口。

设计文档锚点: §4.1 与 §10 集成入口。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

# 让 ``python scripts/train_phase1.py`` 可直接运行。
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.phase1_config import (  # noqa: E402
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
    # reward 对齐 / 模型
    p.add_argument(
        "--reward-alignment",
        choices=["paper_formula", "next_row_execution"],
        default="paper_formula",
    )
    p.add_argument("--num-archetypes", type=int, default=10)
    p.add_argument("--code-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    # 训练
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
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
    dp = DPConfig(horizon=args.horizon, cost_config=cost)
    encoder_input = EncoderInputConfig()
    codebook = CodebookConfig(health=CodebookHealthConfig())
    model = ModelConfig(
        hidden_dim=args.hidden_dim,
        code_dim=args.code_dim,
        num_codes=args.num_archetypes,
        encoder_input=encoder_input,
        codebook=codebook,
    )
    training = TrainingConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        paper_strict_reproduction=args.paper_strict_reproduction,
    )
    strat = StratificationConfig(
        mode=args.stratification_mode,
        diagnostic_pair_batch_id=args.diagnostic_pair_batch_id,
    )
    config = Phase1Config(
        pair=args.pair,
        train_batch_id=args.train_batch_id,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        artifact_root=args.artifact_root,
        horizon=args.horizon,
        num_demos=args.num_demos,
        sampling_strategy=args.sampling_strategy,
        stratification=strat,
        sampling_health=SamplingHealthConfig(),
        no_trade_control=NoTradeControlConfig(),
        no_trade_code_health=NoTradeCodeHealthConfig(),
        data_augmentation=DataAugmentationConfig(),
        dp=dp,
        model=model,
        training=training,
        selection_policy=SelectionPolicyConfig(),
        diagnostics=DiagnosticsConfig(),
        allow_missing_prospective_diagnostic=args.allow_missing_prospective_diagnostic,
        risk_acknowledged_by=args.risk_acknowledged_by,
        expected_sign_off_followup_batch_id=args.expected_sign_off_followup_batch_id,
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
        print(f"[fatal] Phase I 训练终止: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
