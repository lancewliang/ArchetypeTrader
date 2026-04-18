"""全局配置模块 — ArchetypeTrader 超参数管理

支持通过 dataclass 默认值定义所有超参数，并允许命令行参数覆盖。
"""

import argparse
import json
import os
from dataclasses import dataclass, field, fields
from typing import Dict, List

from src.data.feature_pipeline import FIXED_FEATURES, resolve_cycle_features
from src.utils.logger import get_logger
logger = get_logger(__name__)

@dataclass
class Config:
    """ArchetypeTrader 全局配置，包含所有超参数。"""

    # 数据配置
    data_dir: str = "data"
    result_dir: str = "result"
    train_batch_id: str = "default"
    pairs: List[str] = field(default_factory=lambda: [ "AL","ETH"])

    # 特征维度
    cycle_feature_sets: List[str] = field(default_factory=list)

    # MDP 配置
    action_dim: int = 3  # {0: short, 1: flat, 2: long}
    horizon: int = 72  # h = 72 步
    commission_rate: float = 0.00025  # δ = 0.03%（真实佣金率，用于 evaluation）
    dp_commission_rate: float = 0.0008  # DP planner 用 0.1%（高门槛筛选高利润轨迹）
    train_commission_rate: float = 0.0008  # Phase 1/2/3 训练用 0.06%（2× 真实费率，留安全边际）
    max_positions: Dict[str, int] = field(
        default_factory=lambda: { "ETH": 100, "AL": 10}
    )

    # Phase I 配置
    lstm_hidden_dim: int = 128
    latent_dim: int = 16  # z_e 维度
    num_archetypes: int = 10  # K = 10
    vq_beta0: float = 0.25  # 承诺损失系数
    num_trajectories: int = 30000  # 论文 Phase I 默认采样 30k DP trajectories
    phase1_epochs: int = 300
    phase1_sampling_seed: int = 42  # Phase I 轨迹采样随机种子，用于结果复现
    pretrain_epochs: int = 10  # 连续潜在预训练轮数（无 VQ 量化）

    # Phase II 配置
    phase2_hidden_dim: int = 128       # SelectionAgent 共享层宽度
    phase2_bottleneck_dim: int = 64    # SelectionAgent 瓶颈层宽度
    phase2_total_steps: int = 4_000_000
    selection_alpha: float = 1.0  # Phase II 中 imitation/KL 正则的权重；对应 Eq.(5) 的 α，用于平衡“按环境收益优化”与“贴近 DP/VQ 给出的 archetype 标签 â_sel”。值越大，selector 越倾向模仿示范原型选择；值越小，越依赖 PPO 按实际收益自由探索
    phase2_stop_on_unhealthy: bool = False  # 若 Phase II 结束验证不健康则直接退出
    phase2_warmup_steps: int = 800_000  # 热身期步数，只有超过此步数的模型才能成为最优模型
    phase2_rollout_batch_size: int =1024*6  # 每轮 rollout 收集的 horizon 数；越大统计更稳，但单轮更新更慢、更占显存
    phase2_ppo_epochs: int = 16  # 同一批 rollout 数据重复做 PPO 更新的轮数；增大可提高样本利用率，但也更容易过拟合旧策略
    phase2_minibatch_size: int = 1024*3 # 每个 PPO epoch 内的小批量大小；需小于 rollout_batch_size，过大时会退化为近似 full-batch 更新
    phase2_clip_eps: float = 0.2  # PPO ratio 裁剪阈值；限制新旧策略偏移幅度，防止 selector 一次更新跳太远
    phase2_vf_coef: float = 0.001  # Critic 的 value loss 权重；越大越强调 return 拟合，越小越避免 value 分支主导训练
    phase2_ent_coef: float = 0.02  # 策略熵正则权重；鼓励探索更多 archetype，过大可能让选择分布过于发散
    phase2_max_grad_norm: float = 1.0  # 梯度裁剪上限；用于抑制 PPO 更新中的梯度爆炸，稳定 actor/critic 训练
    phase2_log_interval: int = 1000000  # 训练日志输出间隔（按 horizon step 计）；调小更易观测训练过程，但日志会更频繁
    phase2_eval_max_horizons: int | None = None  # 验证时最多评估多少个 horizon；None 表示跑完整验证集，便于最准确对比
    phase2_diagnostic_horizons: int = 256  # 训练诊断阶段抽样的 horizon 数；用于 learned/random/oracle 对比，越大诊断越稳但更耗时

    # Phase III 配置
    phase3_total_steps: int = 1_000_000
    phase3_num_envs: int = 16            # 每轮并行收集的 horizon 数，增大可提升 GPU 利用率
    refinement_hidden_dim: int = 128  # Refinement Agent 隐藏层维度
    refinement_beta1: float = 0.5  # regret 系数，可选 {0.3, 0.5, 0.7}
    refinement_beta2: float = 1.0  # 策略正则化系数
    phase3_clip_eps: float = 0.2  # PPO 裁剪范围
    phase3_ppo_epochs: int = 4  # 每批数据重复训练轮数
    phase3_vf_coef: float = 0.5  # value loss 系数
    phase3_ent_coef: float = 0.01  # 熵正则化系数
    phase3_max_grad_norm: float = 0.5  # 梯度裁剪阈值

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

    @property
    def cycle_features(self) -> List[str]:
        """根据配置选择 short/middle/long cycle 特征。"""
        return resolve_cycle_features(self.cycle_feature_sets)

    @property
    def fixed_feature_dim(self) -> int:
        return len(FIXED_FEATURES)

    @property
    def cycle_feature_dim(self) -> int:
        return len(self.cycle_features)

    @property
    def state_dim(self) -> int:
        """状态维度 = fixed features + selected cycle features。"""
        return self.fixed_feature_dim + self.cycle_feature_dim

    def get_batch_result_dir(self) -> str:
        """批次级结果目录: result/批次号。"""
        return os.path.join(self.result_dir, self.train_batch_id)

    def get_pair_result_dir(self, pair: str) -> str:
        """交易对 + 批次级结果目录: result/品种/批次号。"""
        return os.path.join(self.result_dir, pair, self.train_batch_id)

    def get_stage_result_dir(self, pair: str, stage_name: str) -> str:
        """阶段结果目录: result/品种/批次号/阶段目录。"""
        return os.path.join(self.get_pair_result_dir(pair), stage_name)

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
        "--train-batch-id",
        type=str,
        default=None,
        help="训练批次号，用于隔离并行任务结果目录",
    )
    parser.add_argument(
        "--cycle-feature-sets",
        type=str,
        default=None,
        help="启用的周期特征组，逗号分隔，可选 short,middle,long",
    )
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
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=None,
        help="连续潜在预训练轮数",
    )

    # Phase II
    parser.add_argument(
        "--phase2-total-steps", type=int, default=None, help="Phase II 总训练步数"
    )
    parser.add_argument(
        "--selection-alpha",
        type=float,
        default=None,
        help="Phase II 中 imitation/KL 正则权重；越大越贴近 DP/VQ 的 archetype 标签，越小越偏向按环境收益自由学习",
    )
    parser.add_argument(
        "--phase2-warmup-steps",
        type=int,
        default=None,
        help="Phase II 热身期步数，只有超过此步数的模型才能成为最优模型（默认1500000）",
    )
    parser.add_argument(
        "--phase2-rollout-batch-size",
        type=int,
        default=None,
        help="每轮 PPO rollout 收集的 horizon 数量；越大统计更稳，但更耗时、更占显存",
    )
    parser.add_argument(
        "--phase2-ppo-epochs",
        type=int,
        default=None,
        help="对同一批 rollout 数据重复执行 PPO 更新的轮数",
    )
    parser.add_argument(
        "--phase2-minibatch-size",
        type=int,
        default=None,
        help="每个 PPO epoch 内的小批量大小，通常应小于 rollout_batch_size",
    )
    parser.add_argument(
        "--phase2-clip-eps",
        type=float,
        default=None,
        help="PPO 裁剪阈值 epsilon，用于限制新旧策略概率比的偏移",
    )
    parser.add_argument(
        "--phase2-vf-coef",
        type=float,
        default=None,
        help="PPO 中 critic value loss 的权重系数",
    )
    parser.add_argument(
        "--phase2-ent-coef",
        type=float,
        default=None,
        help="PPO 熵正则系数，用于鼓励 archetype 选择探索",
    )
    parser.add_argument(
        "--phase2-max-grad-norm",
        type=float,
        default=None,
        help="PPO 梯度裁剪阈值，用于稳定 actor/critic 更新",
    )
    parser.add_argument(
        "--phase2-log-interval",
        type=int,
        default=None,
        help="Phase II 日志打印间隔（步数）",
    )
    parser.add_argument(
        "--phase2-eval-max-horizons",
        type=int,
        default=None,
        help="Phase II 验证时最多评估的 horizon 数；不设则评估完整验证集",
    )
    parser.add_argument(
        "--phase2-diagnostic-horizons",
        type=int,
        default=None,
        help="Phase II 训练诊断时抽样的 horizon 数，用于 learned/random/oracle 对比",
    )
    parser.add_argument(
        "--phase2-stop-on-unhealthy",
        action="store_true",
        default=None,
        help="若 Phase II 结束验证健康度非 healthy，则以非 0 状态退出（阻止进入 Phase III）",
    )

    # Phase III
    parser.add_argument(
        "--phase3-total-steps", type=int, default=None, help="Phase III 总训练步数"
    )
    parser.add_argument(
        "--phase3-num-envs", type=int, default=None,
        help="Phase III 每轮并行收集的 horizon 数 (默认 8)",
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

    if args.cycle_feature_sets is not None:
        parsed_cycle_sets = [
            item.strip() for item in args.cycle_feature_sets.split(",") if item.strip()
        ]
        valid_sets = {"short", "middle", "long"}
        invalid_sets = sorted(set(parsed_cycle_sets) - valid_sets)
        if invalid_sets:
            parser.error(
                f"--cycle-feature-sets 包含无效值: {invalid_sets}，可选: {sorted(valid_sets)}"
            )
        args.cycle_feature_sets = parsed_cycle_sets
    else:
        args.cycle_feature_sets = ["middle"]

    if args.train_batch_id is not None:
        args.train_batch_id = args.train_batch_id.strip()
        if not args.train_batch_id:
            parser.error("--train-batch-id 不能为空")
        invalid_separators = [os.sep]
        if os.altsep:
            invalid_separators.append(os.altsep)
        if any(sep in args.train_batch_id for sep in invalid_separators):
            parser.error("--train-batch-id 不能包含路径分隔符")

    # 清理 argparse 添加的 pair 属性（非 Config 字段）
    delattr(args, "pair")

    # 将 kebab-case 属性名映射到 snake_case
    _remap = {
        "data_dir": getattr(args, "data_dir", None),
        "result_dir": getattr(args, "result_dir", None),
        "train_batch_id": getattr(args, "train_batch_id", None),
        "commission_rate": getattr(args, "commission_rate", None),
        "num_archetypes": getattr(args, "num_archetypes", None),
        "num_trajectories": getattr(args, "num_trajectories", None),
        "phase1_epochs": getattr(args, "phase1_epochs", None),
        "latent_dim": getattr(args, "latent_dim", None),
        "vq_beta0": getattr(args, "vq_beta0", None),
        "phase1_sampling_seed": getattr(args, "phase1_sampling_seed", None),
        "pretrain_epochs": getattr(args, "pretrain_epochs", None),
        "phase2_total_steps": getattr(args, "phase2_total_steps", None),
        "selection_alpha": getattr(args, "selection_alpha", None),
        "phase2_warmup_steps": getattr(args, "phase2_warmup_steps", None),
        "phase2_rollout_batch_size": getattr(args, "phase2_rollout_batch_size", None),
        "phase2_ppo_epochs": getattr(args, "phase2_ppo_epochs", None),
        "phase2_minibatch_size": getattr(args, "phase2_minibatch_size", None),
        "phase2_clip_eps": getattr(args, "phase2_clip_eps", None),
        "phase2_vf_coef": getattr(args, "phase2_vf_coef", None),
        "phase2_ent_coef": getattr(args, "phase2_ent_coef", None),
        "phase2_max_grad_norm": getattr(args, "phase2_max_grad_norm", None),
        "phase2_log_interval": getattr(args, "phase2_log_interval", None),
        "phase2_eval_max_horizons": getattr(args, "phase2_eval_max_horizons", None),
        "phase2_diagnostic_horizons": getattr(args, "phase2_diagnostic_horizons", None),
        "phase2_stop_on_unhealthy": getattr(args, "phase2_stop_on_unhealthy", None),
        "phase3_total_steps": getattr(args, "phase3_total_steps", None),
        "phase3_num_envs": getattr(args, "phase3_num_envs", None),
        "batch_size": getattr(args, "batch_size", None),
        "discount_factor": getattr(args, "discount_factor", None),
    }
    for k, v in _remap.items():
        if v is not None:
            setattr(args, k, v)
    logger.info(f"Config: {args}")
    return Config.from_args(args)
