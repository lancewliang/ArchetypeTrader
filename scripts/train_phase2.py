"""Phase II 训练入口。

设计文档锚点: Phase II 执行计划 §Step 1 / §Step 9。

CLI 参数至少包含:
--pair / --phase1-batch-id / --phase2-batch-id / --train-file / --val-file / --test-file /
--total-timesteps / --num-envs / --rollout-length / --seed /
--allow-phase1-hindsight-warning / --paper-strict-reproduction / --resume-from
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
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
    p.add_argument(
        "--max-position",
        type=int,
        default=None,
        help="最大持仓；默认从 Phase I phase1_config.yaml 继承，缺失时回退为 1",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    # KL/demo 消融
    p.add_argument("--kl-demo-coef", type=float, default=None,
                    help="覆盖 PPO 配置中的 kl_demo_coef")
    p.add_argument("--kl-demo-anneal-to", type=float, default=None,
                    help="kl_demo_coef 退火终值")
    p.add_argument("--run-kl-demo-ablation", action="store_true",
                    help="运行 KL/demo alpha 消融矩阵")
    p.add_argument("--kl-demo-ablation-values", nargs="*", type=float,
                    default=[0.0, 0.1, 0.5, 1.0],
                    help="KL/demo 消融 alpha 列表")
    # 控制开关
    p.add_argument("--allow-phase1-hindsight-warning", action="store_true",
                    help="允许 Phase I hindsight_bias_warning=exceeded")
    p.add_argument("--paper-strict-reproduction", action="store_true")
    p.add_argument("--resume-from", default=None,
                    help="从 checkpoint 恢复训练")
    return p


def build_config(args: argparse.Namespace) -> Phase2Config:
    """把 CLI args 翻译为 Phase2Config。"""
    max_position = (
        args.max_position
        if args.max_position is not None
        else _phase1_max_position_or_default(args)
    )
    ppo_kwargs: dict = {
        "update_epochs": args.update_epochs,
        "minibatch_size": args.minibatch_size,
    }
    if args.kl_demo_coef is not None:
        ppo_kwargs["kl_demo_coef"] = args.kl_demo_coef
    if args.kl_demo_anneal_to is not None:
        ppo_kwargs["kl_demo_anneal_to"] = args.kl_demo_anneal_to

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
        max_position=max_position,
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
    if config.paper_strict_reproduction:
        config = config.apply_paper_strict_overrides()
    return config


def _phase1_max_position_or_default(args: argparse.Namespace) -> int:
    """Read Phase I max_position so Phase II defaults to the frozen contract."""
    config_path = (
        Path(args.artifact_root)
        / args.pair
        / args.phase1_batch_id
        / "phase1"
        / "phase1_config.yaml"
    )
    if not config_path.exists():
        return 1
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            phase1_config = yaml.safe_load(f) or {}
        dp_config = phase1_config.get("dp", {})
        max_position = dp_config.get("max_position")
        if max_position is None:
            return 1
        return int(max_position)
    except Exception:
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Phase II 训练主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_config(args)

    if args.run_kl_demo_ablation:
        return run_kl_demo_ablation(config, args.kl_demo_ablation_values)

    trainer = Phase2Trainer(config)
    try:
        trainer.run()
    except Phase2FatalError as exc:
        logger = getattr(trainer, "_logger", None)
        if logger is not None:
            logger.exception("phase2_fatal_error error=%s", exc)
        print(f"[fatal] Phase II 训练终止: {exc}", file=sys.stderr)
        return 1
    except Exception:
        logger = getattr(trainer, "_logger", None)
        if logger is not None:
            logger.exception("phase2_unexpected_error")
        raise
    return 0


def run_kl_demo_ablation(config: Phase2Config, values: Sequence[float]) -> int:
    """运行 KL/demo alpha 消融矩阵并写 summary。

    每个 alpha 使用独立 phase2 batch suffix，避免覆盖主训练产物。
    """
    summary = []
    for alpha in values:
        tag = str(alpha).replace(".", "p")
        ablation_config = replace(
            config,
            phase2_batch_id=f"{config.phase2_batch_id}_kl{tag}",
            ppo=replace(config.ppo, kl_demo_coef=float(alpha)),
        )
        trainer = Phase2Trainer(ablation_config)
        try:
            artifacts = trainer.run()
        except Phase2FatalError as exc:
            logger = getattr(trainer, "_logger", None)
            if logger is not None:
                logger.exception("phase2_ablation_fatal_error alpha=%s error=%s", alpha, exc)
            summary.append({
                "kl_demo_coef": float(alpha),
                "status": "failed",
                "error": str(exc),
            })
            continue
        summary.append({
            "kl_demo_coef": float(alpha),
            "status": "ok",
            "phase2_batch_id": ablation_config.phase2_batch_id,
            "phase2_report": str(artifacts.phase2_report),
        })

    output_dir = config.artifacts_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase2_ablation_kl_demo.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"runs": summary}, f, ensure_ascii=False, indent=2)
    csv_path = output_dir / "phase2_ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["kl_demo_coef", "status", "phase2_batch_id", "phase2_report", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary:
            writer.writerow({key: item.get(key, "") for key in fieldnames})
    print(f"[info] KL/demo ablation summary written to {output_path}")
    return 0 if all(item["status"] == "ok" for item in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
