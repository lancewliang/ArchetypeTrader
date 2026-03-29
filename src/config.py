"""全局配置模块 — ArchetypeTrader 超参数管理

支持通过 dataclass 默认值定义所有超参数，并允许命令行参数覆盖。
"""

import argparse
import json
from dataclasses import dataclass, field, fields
from typing import Dict, List


@dataclass
class Config:
    """ArchetypeTrader 全局配置，包含所有超参数。"""

    # 数据配置
    data_dir: str = "data"
    result_dir: str = "result"
    pairs: List[str] = field(default_factory=lambda: ["ETH"])

    # 特征维度
    single_feature_dim: int = 36
    trend_feature_dim: int = 9
    state_dim: int = 45  # single + trend

    # MDP 配置
    action_dim: int = 3  # {0: short, 1: flat, 2: long}
    horizon: int = 72  # h = 72 步
    commission_rate: float = 0.0002  # δ = 0.02%
    max_positions: Dict[str, int] = field(
        default_factory=lambda: {"BTC": 8, "ETH": 100, "DOT": 2500, "BNB": 200}
    )

    # Phase I 配置
    lstm_hidden_dim: int = 128
    latent_dim: int = 16  # z_e 维度
    num_archetypes: int = 10  # K = 10
    vq_beta0: float = 0.25  # 承诺损失系数
    num_trajectories: int = 30000  # 论文 Phase I 默认采样 30k DP trajectories
    phase1_epochs: int = 100
    phase1_sampling_seed: int = 42  # Phase I 轨迹采样随机种子，用于结果复现

    # Phase II 配置
    phase2_total_steps: int = 8000*100
    selection_alpha: float = 1.0  # KL 惩罚系数

    # Phase III 配置
    phase3_total_steps: int = 1_000_000
    refinement_beta1: float = 0.5  # regret 系数，可选 {0.3, 0.5, 0.7}
    refinement_beta2: float = 1.0  # 策略正则化系数

    # 通用训练配置
    discount_factor: float = 0.99  # γ
    learning_rate: float = 3e-4
    batch_size: int = 256

    # 数据划分
    train_start: str = "2021-06-01"
    train_end: str = "2023-05-31"
    val_start: str = "2023-06-01"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2024-09-01"

    # 评估
    annualization_factor: int = 52560  # 10分钟级别年化因子

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """从已解析的 argparse.Namespace 创建 Config 实例。

        只覆盖用户在命令行中显式提供的参数，其余保持默认值。
        """
        config = cls()
        for f in fields(cls):
            if hasattr(args, f.name):
                val = getattr(args, f.name)
                if val is not None:
                    setattr(config, f.name, val)
        return config


def parse_args(argv: list | None = None) -> Config:
    """解析命令行参数并返回 Config 实例。

    Args:
        argv: 命令行参数列表。None 时使用 sys.argv。

    Returns:
        填充了 CLI 覆盖值的 Config 实例。
    """
    parser = argparse.ArgumentParser(description="ArchetypeTrader 配置")

    # 数据配置
    parser.add_argument("--data-dir", type=str, default=None, help="数据根目录")
    parser.add_argument("--result-dir", type=str, default=None, help="结果输出目录")
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="单个交易对 (BTC/ETH/DOT/BNB)，覆盖 pairs 列表",
    )

    # MDP 配置
    parser.add_argument("--horizon", type=int, default=None, help="交易周期长度")
    parser.add_argument("--commission-rate", type=float, default=None, help="佣金率")

    # Phase I
    parser.add_argument("--num-archetypes", type=int, default=None, help="原型数量 K")
    parser.add_argument(
        "--num-trajectories", type=int, default=None, help="DP 示范轨迹数量"
    )
    parser.add_argument(
        "--phase1-epochs", type=int, default=None, help="Phase I 训练轮数"
    )
    parser.add_argument(
        "--latent-dim", type=int, default=None, help="潜在嵌入维度"
    )
    parser.add_argument(
        "--vq-beta0", type=float, default=None, help="VQ 承诺损失系数"
    )
    parser.add_argument(
        "--phase1-sampling-seed",
        type=int,
        default=None,
        help="Phase I 轨迹采样随机种子",
    )

    # Phase II
    parser.add_argument(
        "--phase2-total-steps", type=int, default=None, help="Phase II 总训练步数"
    )
    parser.add_argument(
        "--selection-alpha", type=float, default=None, help="KL 惩罚系数"
    )

    # Phase III
    parser.add_argument(
        "--phase3-total-steps", type=int, default=None, help="Phase III 总训练步数"
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=None,
        dest="refinement_beta1",
        help="Regret 系数 β₁ (0.3/0.5/0.7)",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=None,
        dest="refinement_beta2",
        help="策略正则化系数 β₂",
    )

    # 通用训练
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate", help="学习率")
    parser.add_argument("--batch-size", type=int, default=None, help="批量大小")
    parser.add_argument(
        "--discount-factor", type=float, default=None, help="折扣因子 γ"
    )

    args = parser.parse_args(argv)

    # --pair 覆盖 pairs 列表为单元素
    if args.pair is not None:
        args.pairs = [args.pair]
    else:
        args.pairs = None

    # 清理 argparse 添加的 pair 属性（非 Config 字段）
    delattr(args, "pair")

    # 将 kebab-case 属性名映射到 snake_case
    _remap = {
        "data_dir": getattr(args, "data_dir", None),
        "result_dir": getattr(args, "result_dir", None),
        "commission_rate": getattr(args, "commission_rate", None),
        "num_archetypes": getattr(args, "num_archetypes", None),
        "num_trajectories": getattr(args, "num_trajectories", None),
        "phase1_epochs": getattr(args, "phase1_epochs", None),
        "latent_dim": getattr(args, "latent_dim", None),
        "vq_beta0": getattr(args, "vq_beta0", None),
        "phase1_sampling_seed": getattr(args, "phase1_sampling_seed", None),
        "phase2_total_steps": getattr(args, "phase2_total_steps", None),
        "selection_alpha": getattr(args, "selection_alpha", None),
        "phase3_total_steps": getattr(args, "phase3_total_steps", None),
        "batch_size": getattr(args, "batch_size", None),
        "discount_factor": getattr(args, "discount_factor", None),
    }
    for k, v in _remap.items():
        if v is not None:
            setattr(args, k, v)

    return Config.from_args(args)
