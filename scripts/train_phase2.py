"""Phase II 训练入口。

设计文档锚点: Phase II 执行计划 §Step 1 / §Step 9。

CLI 参数至少包含:
--pair / --phase1-batch-id / --phase2-batch-id / --train-file / --val-file / --test-file /
--total-timesteps / --num-envs / --rollout-length / --seed /
--allow-phase1-hindsight-warning / --paper-strict-reproduction / --resume-from
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.phase2_config import (  # noqa: E402
    Phase1ArtifactsConfig,
    Phase2Config,
    PPOConfig,
)
from src.trainers.phase2_trainer import Phase2FatalError, Phase2Trainer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    p = argparse.ArgumentParser(description="ArchetypeTrader Phase II 训练入口")
    # 路径
    p.add_argument("--pair", required=True, help="交易标的")
    p.add_argument("--phase1-batch-id", required=True, help="Phase I batch ID")
    p.add_argument("--phase2-batch-id", required=True, help="Phase II batch ID")
    p.add_argument("--train-file", required=True, help="训练数据 feather 路径")
    p.add_argument("--val-file", required=True, help="验证数据 feather 路径")
    p.add_argument("--test-file", required=True, help="测试数据 feather 路径")
    p.add_argument("--artifact-root", default="artifacts")
    # 训练超参
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--rollout-length", type=int, default=128)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    # KL/demo 消融
    p.add_argument("--kl-demo-coef", type=float, default=None,
                    help="覆盖 PPO 配置中的 kl_demo_coef")
    p.add_argument("--kl-demo-anneal-to", type=float, default=None,
                    help="kl_demo_coef 退火终值")
    # 控制开关
    p.add_argument("--allow-phase1-hindsight-warning", action="store_true",
                    help="允许 Phase I hindsight_bias_warning=exceeded")
    p.add_argument("--paper-strict-reproduction", action="store_true")
    p.add_argument("--resume-from", default=None,
                    help="从 checkpoint 恢复训练")
    return p


def build_config(args: argparse.Namespace) -> Phase2Config:
    """把 CLI args 翻译为 Phase2Config。"""
    ppo_kwargs: dict = {
        "update_epochs": args.update_epochs,
        "minibatch_size": args.minibatch_size,
    }
    if args.kl_demo_coef is not None:
        ppo_kwargs["kl_demo_coef"] = args.kl_demo_coef

    ppo = PPOConfig(**ppo_kwargs)

    config = Phase2Config(
        pair=args.pair,
        phase1_batch_id=args.phase1_batch_id,
        phase2_batch_id=args.phase2_batch_id,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        artifact_root=args.artifact_root,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        seed=args.seed,
        device=args.device,
        allow_phase1_hindsight_warning=args.allow_phase1_hindsight_warning,
        paper_strict_reproduction=args.paper_strict_reproduction,
        resume_from=args.resume_from,
        ppo=ppo,
        phase1_artifacts=Phase1ArtifactsConfig(
            artifact_root=args.artifact_root,
            pair=args.pair,
            phase1_batch_id=args.phase1_batch_id,
        ),
    )
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Phase II 训练主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_config(args)
    trainer = Phase2Trainer(config)
    try:
        trainer.run()
    except Phase2FatalError as exc:
        print(f"[fatal] Phase II 训练终止: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
