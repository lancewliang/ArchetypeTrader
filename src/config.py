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
    commission_rate: float = 0.0002  # δ = 0.03%（真实佣金率，用于 evaluation）
    dp_commission_rate: float = 0.0008  # DP planner 用 0.1%（高门槛筛选高利润轨迹）
    train_commission_rate: float = 0.0008  # Phase 1/2/3 训练用 0.06%（2× 真实费率，留安全边际）
    max_positions: Dict[str, int] = field(
        default_factory=lambda: { "ETH": 100, "AL": 10}
    )

    # Phase I 配置
    lstm_hidden_dim: int = 256
    latent_dim: int = 32  # z_e 维度
    num_archetypes: int = 10  # K = 10
    vq_beta0: float = 0.25  # 承诺损失系数
    num_trajectories: int = 60000  # 默认 90k（论文 Phase I 为 30k DP trajectories）
    phase1_epochs: int = 500
    phase1_sampling_seed: int = 42  # Phase I 轨迹采样随机种子，用于结果复现
    phase1_start_sampling_mode: str = "hybrid_stratified_importance"  # 起点采样: uniform / stratified / hybrid_stratified_importance
    phase1_stratified_ratio: float = 0.95  # 混合采样中分层随机占比；纯 stratified 模式下作为元数据保留
    phase1_importance_ratio: float = 0.05  # 混合采样中重要性采样占比；纯 stratified 模式下默认关闭
    phase1_sampling_strata: int = 4  # 分层采样分位桶数（波动率 x 趋势）
    phase1_importance_vol_weight: float = 0.8  # 重要性打分中波动率权重
    phase1_importance_net_weight: float = 0.2  # 重要性打分中净收益代理权重
    phase1_usage_profit_alignment_weight: float = 0.01  # 轻量收益-使用率对齐，修正 assignment 而不主导训练
    phase1_usage_profit_alignment_target_corr: float = 0.02  # 仅要求弱正相关，避免再次把收益谱系压平
    phase1_usage_profit_alignment_temperature: float = 0.35  # soft assignment 温度
    phase1_return_aux_weight: float = 0.10  # 收益分桶辅助目标权重
    phase1_return_aux_hidden_dim: int = 64  # 收益分桶头隐藏层宽度
    phase1_return_num_buckets: int = 5  # 收益分桶数量（按轨迹总收益分位数切分）
    phase1_return_soft_assignment_weight: float = 0.50  # 收益分桶 loss 中 soft-assignment 路径占比
    phase1_codebook_separation_weight: float = 0.02  # codebook 分离正则权重
    phase1_codebook_separation_margin: float = 0.35  # cosine 相似度超过该阈值后开始惩罚
    phase1_profit_init_top_ratio: float = 0.25  # 初始化时优先使用高收益样本的 top 比例
    phase1_profit_init_code_ratio: float = 0.50  # 初始化时每个方向分配给高收益子集的 code 比例
    phase1_profit_reset_top_ratio: float = 0.25  # 死码重置时优先抽取高收益样本的 top 比例
    phase1_selection_min_realizable_proxy_return_mean: float = 0.0  # Phase I checkpoint 选择的 realizable proxy 绝对门槛
    phase1_selection_min_realizable_proxy_to_oracle_ratio: float = 0.40  # realizable proxy / oracle 的最低占比
    phase1_selection_min_best_fixed_archetype_return_mean: float = 0.0  # 最佳固定原型收益绝对门槛
    phase1_selection_min_best_fixed_to_oracle_ratio: float = 0.50  # 最佳固定原型收益 / oracle 的最低占比
    phase1_selection_min_return_usage_correlation: float = 0.0  # 高收益原型应至少不被负向压制
    phase1_selection_require_gated_candidate: bool = True  # 若所有候选都不满足 profit gate，则直接报错而不是回退
    pretrain_epochs: int = 10  # 连续潜在预训练轮数（无 VQ 量化）

    # Phase II 配置
    phase2_hidden_dim: int = 256       # SelectionAgent 共享层宽度
    phase2_bottleneck_dim: int = 128    # SelectionAgent 瓶颈层宽度
    phase2_total_steps: int = 1_000_000
    selection_alpha: float = 0.5  # KL / imitation 初始惩罚系数
    phase2_alpha_schedule: str = "linear"  # selection_alpha 调度: constant / linear
    phase2_alpha_final_ratio: float = 0.0  # 线性调度结束时 alpha = initial_alpha × ratio
    phase2_imitation_min_raw_return: float = 0.0  # 仅对 raw horizon return 超过该阈值的样本施加 imitation
    phase2_val_interval_multiplier: int = 10  # 每遍历多少轮 train horizons 做一次验证
    phase2_stop_on_unhealthy: bool = False  # 若 Phase II 结束验证不健康则直接退出
    phase2_rollout_batch_size: int = 1024*2
    phase2_ppo_epochs: int = 8
    phase2_minibatch_size: int = 1024*1
    phase2_clip_eps: float = 0.2
    phase2_vf_coef: float = 0.001
    phase2_ent_coef: float = 0.02
    phase2_max_grad_norm: float = 1.0
    phase2_log_interval: int = 1000000
    phase2_eval_max_horizons: int | None = None
    phase2_diagnostic_horizons: int = 512

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
        "--lstm-hidden-dim",
        type=int,
        default=None,
        help="Phase I encoder/decoder LSTM 隐藏层维度",
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
        "--phase1-start-sampling-mode",
        type=str,
        default=None,
        choices=["uniform", "stratified", "hybrid_stratified_importance"],
        help="Phase I 轨迹起点采样模式",
    )
    parser.add_argument(
        "--phase1-stratified-ratio",
        type=float,
        default=None,
        help="混合采样中分层随机占比",
    )
    parser.add_argument(
        "--phase1-importance-ratio",
        type=float,
        default=None,
        help="混合采样中重要性采样占比",
    )
    parser.add_argument(
        "--phase1-sampling-strata",
        type=int,
        default=None,
        help="分层采样分位桶数（>=2）",
    )
    parser.add_argument(
        "--phase1-importance-vol-weight",
        type=float,
        default=None,
        help="重要性打分中波动率权重（非负）",
    )
    parser.add_argument(
        "--phase1-importance-net-weight",
        type=float,
        default=None,
        help="重要性打分中净收益代理权重（非负）",
    )
    parser.add_argument(
        "--phase1-usage-profit-alignment-weight",
        type=float,
        default=None,
        help="收益-使用率对齐正则权重",
    )
    parser.add_argument(
        "--phase1-usage-profit-alignment-target-corr",
        type=float,
        default=None,
        help="收益-使用率对齐正则的目标相关系数",
    )
    parser.add_argument(
        "--phase1-usage-profit-alignment-temperature",
        type=float,
        default=None,
        help="收益-使用率对齐所用 soft assignment 温度",
    )
    parser.add_argument(
        "--phase1-return-aux-weight",
        type=float,
        default=None,
        help="Phase I 轨迹收益辅助头权重",
    )
    parser.add_argument(
        "--phase1-return-aux-hidden-dim",
        type=int,
        default=None,
        help="Phase I 收益分桶头隐藏层宽度",
    )
    parser.add_argument(
        "--phase1-return-num-buckets",
        type=int,
        default=None,
        help="Phase I 收益分桶数量（>=2）",
    )
    parser.add_argument(
        "--phase1-return-soft-assignment-weight",
        type=float,
        default=None,
        help="Phase I 收益分桶 loss 中 soft-assignment 路径占比",
    )
    parser.add_argument(
        "--phase1-codebook-separation-weight",
        type=float,
        default=None,
        help="codebook 分离正则权重",
    )
    parser.add_argument(
        "--phase1-codebook-separation-margin",
        type=float,
        default=None,
        help="codebook cosine 相似度分离边界",
    )
    parser.add_argument(
        "--phase1-profit-init-top-ratio",
        type=float,
        default=None,
        help="Phase I 初始化时优先采用高收益样本的 top 比例",
    )
    parser.add_argument(
        "--phase1-profit-init-code-ratio",
        type=float,
        default=None,
        help="Phase I 初始化时分配给高收益子集的 code 比例",
    )
    parser.add_argument(
        "--phase1-profit-reset-top-ratio",
        type=float,
        default=None,
        help="Phase I 死码重置时优先采用高收益样本的 top 比例",
    )
    parser.add_argument(
        "--phase1-selection-min-realizable-proxy-return-mean",
        type=float,
        default=None,
        help="Phase I checkpoint 选择时 realizable proxy return mean 的绝对下限",
    )
    parser.add_argument(
        "--phase1-selection-min-realizable-proxy-to-oracle-ratio",
        type=float,
        default=None,
        help="Phase I checkpoint 选择时 realizable proxy / oracle 的最低占比",
    )
    parser.add_argument(
        "--phase1-selection-min-best-fixed-archetype-return-mean",
        type=float,
        default=None,
        help="Phase I checkpoint 选择时最佳固定 archetype 收益的绝对下限",
    )
    parser.add_argument(
        "--phase1-selection-min-best-fixed-to-oracle-ratio",
        type=float,
        default=None,
        help="Phase I checkpoint 选择时最佳固定 archetype / oracle 的最低占比",
    )
    parser.add_argument(
        "--phase1-selection-min-return-usage-correlation",
        type=float,
        default=None,
        help="Phase I checkpoint 选择时收益-使用率相关系数的最低阈值",
    )
    parser.add_argument(
        "--phase1-selection-require-gated-candidate",
        action="store_true",
        default=None,
        help="若没有候选满足 Phase I profit gate，则直接报错而不是回退到原排序",
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
        "--phase2-hidden-dim",
        type=int,
        default=None,
        help="Phase II SelectionAgent 共享层宽度",
    )
    parser.add_argument(
        "--phase2-bottleneck-dim",
        type=int,
        default=None,
        help="Phase II SelectionAgent 瓶颈层宽度",
    )
    parser.add_argument(
        "--selection-alpha", type=float, default=None, help="KL 惩罚系数"
    )
    parser.add_argument(
        "--phase2-alpha-schedule",
        type=str,
        default=None,
        choices=["constant", "linear"],
        help="Phase II selection_alpha 调度方式",
    )
    parser.add_argument(
        "--phase2-alpha-final-ratio",
        type=float,
        default=None,
        help="线性 alpha 调度结束时相对初始值的比例",
    )
    parser.add_argument(
        "--phase2-imitation-min-raw-return",
        type=float,
        default=None,
        help="仅对 raw horizon return 超过该阈值的样本施加 imitation",
    )
    parser.add_argument(
        "--phase2-val-interval-multiplier",
        type=int,
        default=None,
        help="Phase II 每遍历多少轮 train horizons 做一次验证",
    )
    parser.add_argument(
        "--phase2-rollout-batch-size",
        type=int,
        default=None,
        help="每次 PPO rollout 采样的 horizon 数量",
    )
    parser.add_argument(
        "--phase2-ppo-epochs",
        type=int,
        default=None,
        help="每个 rollout 的 PPO 更新轮数",
    )
    parser.add_argument(
        "--phase2-minibatch-size",
        type=int,
        default=None,
        help="PPO 更新时的 minibatch 大小",
    )
    parser.add_argument(
        "--phase2-clip-eps",
        type=float,
        default=None,
        help="PPO 裁剪阈值 epsilon",
    )
    parser.add_argument(
        "--phase2-vf-coef",
        type=float,
        default=None,
        help="PPO value loss 系数",
    )
    parser.add_argument(
        "--phase2-ent-coef",
        type=float,
        default=None,
        help="PPO 熵正则系数",
    )
    parser.add_argument(
        "--phase2-max-grad-norm",
        type=float,
        default=None,
        help="PPO 梯度裁剪阈值",
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
        help="Phase II 验证时最多评估的 horizon 数",
    )
    parser.add_argument(
        "--phase2-diagnostic-horizons",
        type=int,
        default=None,
        help="Phase II 训练诊断采样的 horizon 数",
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

    if args.phase1_stratified_ratio is not None and args.phase1_stratified_ratio < 0:
        parser.error("--phase1-stratified-ratio 必须 >= 0")
    if args.phase1_importance_ratio is not None and args.phase1_importance_ratio < 0:
        parser.error("--phase1-importance-ratio 必须 >= 0")
    if args.phase1_sampling_strata is not None and args.phase1_sampling_strata < 2:
        parser.error("--phase1-sampling-strata 必须 >= 2")
    for name in ["latent_dim", "lstm_hidden_dim", "phase2_hidden_dim", "phase2_bottleneck_dim"]:
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} 必须 >= 1")
    if args.phase1_importance_vol_weight is not None and args.phase1_importance_vol_weight < 0:
        parser.error("--phase1-importance-vol-weight 必须 >= 0")
    if args.phase1_importance_net_weight is not None and args.phase1_importance_net_weight < 0:
        parser.error("--phase1-importance-net-weight 必须 >= 0")
    if args.phase1_usage_profit_alignment_weight is not None and args.phase1_usage_profit_alignment_weight < 0:
        parser.error("--phase1-usage-profit-alignment-weight 必须 >= 0")
    if (
        args.phase1_usage_profit_alignment_target_corr is not None
        and not -1.0 <= args.phase1_usage_profit_alignment_target_corr <= 1.0
    ):
        parser.error("--phase1-usage-profit-alignment-target-corr 必须在 [-1, 1] 范围内")
    if (
        args.phase1_usage_profit_alignment_temperature is not None
        and args.phase1_usage_profit_alignment_temperature <= 0
    ):
        parser.error("--phase1-usage-profit-alignment-temperature 必须 > 0")
    if args.phase1_return_aux_weight is not None and args.phase1_return_aux_weight < 0:
        parser.error("--phase1-return-aux-weight 必须 >= 0")
    if args.phase1_return_aux_hidden_dim is not None and args.phase1_return_aux_hidden_dim < 1:
        parser.error("--phase1-return-aux-hidden-dim 必须 >= 1")
    if args.phase1_return_num_buckets is not None and args.phase1_return_num_buckets < 2:
        parser.error("--phase1-return-num-buckets 必须 >= 2")
    if (
        args.phase1_return_soft_assignment_weight is not None
        and not 0.0 <= args.phase1_return_soft_assignment_weight <= 1.0
    ):
        parser.error("--phase1-return-soft-assignment-weight 必须在 [0, 1] 范围内")
    if args.phase1_codebook_separation_weight is not None and args.phase1_codebook_separation_weight < 0:
        parser.error("--phase1-codebook-separation-weight 必须 >= 0")
    if (
        args.phase1_codebook_separation_margin is not None
        and not -1.0 <= args.phase1_codebook_separation_margin <= 1.0
    ):
        parser.error("--phase1-codebook-separation-margin 必须在 [-1, 1] 范围内")
    for name in [
        "phase1_profit_init_top_ratio",
        "phase1_profit_init_code_ratio",
        "phase1_profit_reset_top_ratio",
    ]:
        value = getattr(args, name, None)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} 必须在 [0, 1] 范围内")
    if (
        args.phase1_selection_min_realizable_proxy_to_oracle_ratio is not None
        and args.phase1_selection_min_realizable_proxy_to_oracle_ratio < 0
    ):
        parser.error("--phase1-selection-min-realizable-proxy-to-oracle-ratio 必须 >= 0")
    if (
        args.phase1_selection_min_best_fixed_to_oracle_ratio is not None
        and args.phase1_selection_min_best_fixed_to_oracle_ratio < 0
    ):
        parser.error("--phase1-selection-min-best-fixed-to-oracle-ratio 必须 >= 0")
    if (
        args.phase1_selection_min_return_usage_correlation is not None
        and not -1.0 <= args.phase1_selection_min_return_usage_correlation <= 1.0
    ):
        parser.error("--phase1-selection-min-return-usage-correlation 必须在 [-1, 1] 范围内")
    if args.selection_alpha is not None and args.selection_alpha < 0:
        parser.error("--selection-alpha 必须 >= 0")
    if (
        args.phase2_alpha_final_ratio is not None
        and args.phase2_alpha_final_ratio < 0
    ):
        parser.error("--phase2-alpha-final-ratio 必须 >= 0")
    if (
        args.phase2_val_interval_multiplier is not None
        and args.phase2_val_interval_multiplier < 1
    ):
        parser.error("--phase2-val-interval-multiplier 必须 >= 1")

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
        "phase1_start_sampling_mode": getattr(args, "phase1_start_sampling_mode", None),
        "phase1_stratified_ratio": getattr(args, "phase1_stratified_ratio", None),
        "phase1_importance_ratio": getattr(args, "phase1_importance_ratio", None),
        "phase1_sampling_strata": getattr(args, "phase1_sampling_strata", None),
        "phase1_importance_vol_weight": getattr(args, "phase1_importance_vol_weight", None),
        "phase1_importance_net_weight": getattr(args, "phase1_importance_net_weight", None),
        "phase1_usage_profit_alignment_weight": getattr(args, "phase1_usage_profit_alignment_weight", None),
        "phase1_usage_profit_alignment_target_corr": getattr(
            args, "phase1_usage_profit_alignment_target_corr", None,
        ),
        "phase1_usage_profit_alignment_temperature": getattr(
            args, "phase1_usage_profit_alignment_temperature", None,
        ),
        "phase1_return_aux_weight": getattr(args, "phase1_return_aux_weight", None),
        "phase1_return_aux_hidden_dim": getattr(args, "phase1_return_aux_hidden_dim", None),
        "phase1_return_num_buckets": getattr(args, "phase1_return_num_buckets", None),
        "phase1_return_soft_assignment_weight": getattr(
            args, "phase1_return_soft_assignment_weight", None,
        ),
        "phase1_codebook_separation_weight": getattr(args, "phase1_codebook_separation_weight", None),
        "phase1_codebook_separation_margin": getattr(args, "phase1_codebook_separation_margin", None),
        "phase1_profit_init_top_ratio": getattr(args, "phase1_profit_init_top_ratio", None),
        "phase1_profit_init_code_ratio": getattr(args, "phase1_profit_init_code_ratio", None),
        "phase1_profit_reset_top_ratio": getattr(args, "phase1_profit_reset_top_ratio", None),
        "phase1_selection_min_realizable_proxy_return_mean": getattr(
            args, "phase1_selection_min_realizable_proxy_return_mean", None,
        ),
        "phase1_selection_min_realizable_proxy_to_oracle_ratio": getattr(
            args, "phase1_selection_min_realizable_proxy_to_oracle_ratio", None,
        ),
        "phase1_selection_min_best_fixed_archetype_return_mean": getattr(
            args, "phase1_selection_min_best_fixed_archetype_return_mean", None,
        ),
        "phase1_selection_min_best_fixed_to_oracle_ratio": getattr(
            args, "phase1_selection_min_best_fixed_to_oracle_ratio", None,
        ),
        "phase1_selection_min_return_usage_correlation": getattr(
            args, "phase1_selection_min_return_usage_correlation", None,
        ),
        "phase1_selection_require_gated_candidate": getattr(
            args, "phase1_selection_require_gated_candidate", None,
        ),
        "pretrain_epochs": getattr(args, "pretrain_epochs", None),
        "phase2_total_steps": getattr(args, "phase2_total_steps", None),
        "selection_alpha": getattr(args, "selection_alpha", None),
        "phase2_alpha_schedule": getattr(args, "phase2_alpha_schedule", None),
        "phase2_alpha_final_ratio": getattr(args, "phase2_alpha_final_ratio", None),
        "phase2_imitation_min_raw_return": getattr(args, "phase2_imitation_min_raw_return", None),
        "phase2_val_interval_multiplier": getattr(args, "phase2_val_interval_multiplier", None),
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
