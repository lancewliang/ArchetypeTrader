"""Phase II 训练入口。

默认从 ``artifacts/<PAIR>/<TRAIN_BATCH_ID>/phase1/checkpoints/best_checkpoint.pt``
读取 Phase I best checkpoint，并将 Phase II 产物写入同一批次目录。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1.phase1_main import Phase1MainConfig  # noqa: E402
from src.phase2.phase2_config import (  # noqa: E402
    Phase2DatasetConfig,
    Phase2MainConfig,
    Phase2RewardConfig,
    Phase2TrainConfig,
)
from src.phase2.phase2_main import Phase2MainFlow  # noqa: E402
from src.utils import RuntimeUtils  # noqa: E402


DEFAULT_PAIR = "FU"
DEFAULT_TRAIN_BATCH_ID = "batch_005"
DEFAULT_SEED = 42
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_DETERMINISTIC = True


def parse_args() -> argparse.Namespace:
    """解析 Phase II 训练参数。"""

    parser = argparse.ArgumentParser(description="Train Phase II archetype selector.")
    parser.add_argument("--pair", default=DEFAULT_PAIR, help="交易标的目录名。")
    parser.add_argument(
        "--train-batch-id",
        "--batchid",
        dest="train_batch_id",
        default=DEFAULT_TRAIN_BATCH_ID,
        help="训练批次 ID。",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="产物根目录。",
    )
    parser.add_argument(
        "--phase1-checkpoint-path",
        type=Path,
        default=None,
        help="Phase I best checkpoint 路径；默认使用当前 pair/batch 下的 best checkpoint。",
    )
    parser.add_argument("--device", default="cuda", help="训练设备，例如 cuda 或 cpu。")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子。")
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help="日志级别，例如 INFO 或 DEBUG。",
    )
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="关闭 PyTorch deterministic 设置。",
    )

    parser.add_argument("--tsize", type=int, default=Phase2DatasetConfig().tsize)
    parser.add_argument("--epochs", type=int, default=Phase2TrainConfig().epochs)
    parser.add_argument("--batch-size", type=int, default=Phase2TrainConfig().batch_size)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=Phase2TrainConfig().learning_rate,
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=Phase2TrainConfig().replay_capacity,
    )
    parser.add_argument(
        "--learning-start-epoch",
        type=int,
        default=Phase2TrainConfig().learning_start_epoch,
    )
    parser.add_argument(
        "--updates-per-epoch",
        type=int,
        default=Phase2TrainConfig().updates_per_epoch,
    )
    parser.add_argument(
        "--target-update-interval-epochs",
        type=int,
        default=Phase2TrainConfig().target_update_interval_epochs,
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=Phase2TrainConfig().epsilon_start,
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=Phase2TrainConfig().epsilon_end,
    )
    parser.add_argument(
        "--epsilon-decay-epochs",
        type=int,
        default=Phase2TrainConfig().epsilon_decay_epochs,
    )
    parser.add_argument(
        "--td-loss-beta",
        type=float,
        default=Phase2TrainConfig().td_loss_beta,
    )
    parser.add_argument(
        "--imitation-loss-beta",
        type=float,
        default=Phase2TrainConfig().imitation_loss_beta,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=Phase2TrainConfig().max_grad_norm,
    )

    parser.add_argument("--gamma", type=float, default=Phase2RewardConfig().gamma)
    parser.add_argument("--fee-rate", type=float, default=Phase2RewardConfig().fee_rate)
    parser.add_argument(
        "--imitation-alpha",
        type=float,
        default=Phase2RewardConfig().imitation_alpha,
    )
    parser.add_argument(
        "--reward-clip",
        type=float,
        default=Phase2RewardConfig().reward_clip,
    )
    parser.add_argument(
        "--no-normalize-rewards",
        action="store_true",
        help="关闭 replay batch reward 标准化。",
    )

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"[phase2] ignored unknown args: {' '.join(unknown_args)}")
    return args


def default_phase1_checkpoint_path(args: argparse.Namespace) -> Path:
    """返回当前 pair/batch 下的 Phase I best checkpoint 默认路径。"""

    return (
        args.artifacts_root
        / args.pair
        / args.train_batch_id
        / "phase1"
        / "checkpoints"
        / "best_checkpoint.pt"
    )


def initialize_runtime(args: argparse.Namespace) -> None:
    """初始化入口脚本层面的日志和随机种子。"""

    output_dir = args.artifacts_root / args.pair / args.train_batch_id / "phase2"
    logger = RuntimeUtils.init_logging(
        name="archetype_trader.phase2",
        log_file=output_dir / "phase2.log",
        level=args.log_level,
    )
    seed_status = RuntimeUtils.init_random_seed(
        args.seed,
        deterministic=not args.non_deterministic and DEFAULT_DETERMINISTIC,
    )
    logger.info("runtime initialized: seed_status=%s", seed_status)


def create_phase2_flow(args: argparse.Namespace) -> Phase2MainFlow:
    """使用 CLI 参数创建 Phase II 主流程实例。"""

    phase1_checkpoint_path = args.phase1_checkpoint_path
    if phase1_checkpoint_path is None:
        phase1_checkpoint_path = default_phase1_checkpoint_path(args)

    phase1_config = Phase1MainConfig(
        pair=args.pair,
        train_batch_id=args.train_batch_id,
        device=args.device,
    )
    main_config = Phase2MainConfig(
        pair=args.pair,
        train_batch_id=args.train_batch_id,
        phase1_checkpoint_path=phase1_checkpoint_path,
        artifacts_root=args.artifacts_root,
        device=args.device,
    )
    dataset_config = Phase2DatasetConfig(tsize=args.tsize)
    train_config = replace(
        Phase2TrainConfig(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        replay_capacity=args.replay_capacity,
        learning_start_epoch=args.learning_start_epoch,
        updates_per_epoch=args.updates_per_epoch,
        target_update_interval_epochs=args.target_update_interval_epochs,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_epochs=args.epsilon_decay_epochs,
        td_loss_beta=args.td_loss_beta,
        imitation_loss_beta=args.imitation_loss_beta,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )
    reward_config = replace(
        Phase2RewardConfig(),
        gamma=args.gamma,
        fee_rate=args.fee_rate,
        imitation_alpha=args.imitation_alpha,
        reward_clip=args.reward_clip,
        normalize_rewards=not args.no_normalize_rewards,
    )
    return Phase2MainFlow(
        main_config,
        phase1_config,
        dataset_config=dataset_config,
        train_config=train_config,
        reward_config=reward_config,
    )


def main() -> None:
    """创建并运行 Phase II 主流程。"""

    args = parse_args()
    initialize_runtime(args)
    flow = create_phase2_flow(args)
    flow.run()


if __name__ == "__main__":
    main()
