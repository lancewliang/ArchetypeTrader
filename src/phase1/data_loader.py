"""Phase I 数据加载模块

本模块提供 Phase I 训练中的数据加载和环境构建功能。

Functions:
    load_data_and_env: 加载特征数据并初始化 TradingEnv
    prepare_trajectory_dataset: 检查缓存/生成 DP 示范轨迹
    build_val_env: 构建验证集环境
    inspect_trajectory_cache: 检查轨迹缓存兼容性
    backup_incompatible_cache: 备份不兼容的缓存
    log_training_data_scale: 记录训练数据规模
    expected_num_available_starts: 计算可用起点数量
"""

import os
import shutil
from datetime import datetime
from typing import Any, List, Tuple

import numpy as np

from src.data.dataset import TrajectoryDataset
from src.data.feature_pipeline import FeaturePipeline
from src.env.trading_env import TradingEnv
from src.phase1.dp_planner import DPPlanner
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 论文参考训练数据规模
PAPER_PHASE1_REFERENCE_TRAIN_ROWS = 1_400_000


def expected_num_available_starts(total_rows: int, horizon: int) -> int:
    """计算滑窗采样协议下全部合法起点数量。

    Args:
        total_rows: 总数据行数
        horizon: horizon 长度

    Returns:
        可用起点数量
    """
    return max(total_rows - horizon + 1, 0)


def log_training_data_scale(train_rows: int) -> None:
    """记录当前训练数据规模与论文规模的差异。

    Args:
        train_rows: 当前训练集行数
    """
    ratio = float(train_rows) / float(PAPER_PHASE1_REFERENCE_TRAIN_ROWS)
    logger.warning(
        "当前训练集行数=%d，论文约使用=%d 行；当前约为论文数据规模的 %.2f%%。"
        "这仍属于严格论文算法/公式下的 reduced-data reproduction，而非同数据规模复现。",
        train_rows,
        PAPER_PHASE1_REFERENCE_TRAIN_ROWS,
        ratio * 100.0,
    )


def inspect_trajectory_cache(
    traj_path: str,
    config: Any,
    pair: str,
    train_rows: int,
) -> Tuple[bool, List[str]]:
    """检查现有轨迹缓存是否与当前严格论文设置兼容。

    Args:
        traj_path: 轨迹缓存文件路径
        config: 配置对象
        pair: 交易对
        train_rows: 训练数据行数

    Returns:
        (是否兼容, 不兼容原因列表)
    """
    if not os.path.exists(traj_path):
        return False, ["trajectory cache 文件不存在"]

    reasons: List[str] = []
    expected_starts = expected_num_available_starts(train_rows, config.horizon)
    ratio_sum = float(config.phase1_stratified_ratio) + float(config.phase1_importance_ratio)
    if ratio_sum <= 0:
        expected_stratified_ratio = 1.0
        expected_importance_ratio = 0.0
    else:
        expected_stratified_ratio = float(config.phase1_stratified_ratio) / ratio_sum
        expected_importance_ratio = float(config.phase1_importance_ratio) / ratio_sum

    weight_sum = float(config.phase1_importance_vol_weight) + float(config.phase1_importance_net_weight)
    if weight_sum <= 0:
        expected_importance_vol_weight = 0.5
        expected_importance_net_weight = 0.5
    else:
        expected_importance_vol_weight = float(config.phase1_importance_vol_weight) / weight_sum
        expected_importance_net_weight = float(config.phase1_importance_net_weight) / weight_sum

    expected_values = {
        "pair": pair,
        "horizon": int(config.horizon),
        "gamma": float(config.discount_factor),
        "num_sampled_trajectories": int(config.num_trajectories),
        "sampling_seed": int(config.phase1_sampling_seed),
        "sampling_mode": str(config.phase1_start_sampling_mode),
        "sampling_stratified_ratio": float(expected_stratified_ratio),
        "sampling_importance_ratio": float(expected_importance_ratio),
        "sampling_num_strata": int(config.phase1_sampling_strata),
        "sampling_importance_vol_weight": float(expected_importance_vol_weight),
        "sampling_importance_net_weight": float(expected_importance_net_weight),
        "num_available_starts": int(expected_starts),
        "training_rows": int(train_rows),
        "state_dim": int(config.state_dim),
        "commission_rate": float(config.dp_commission_rate),
        "max_position": int(config.max_positions[pair]),
        "algorithm_variant": "paper_single_change",
    }

    with np.load(traj_path, allow_pickle=False) as data:
        required_keys = set(expected_values.keys()) | {"states", "actions", "rewards"}
        missing_keys = sorted(required_keys - set(data.files))
        if missing_keys:
            reasons.append(f"cache 缺少关键元数据: {missing_keys}")
            return False, reasons

        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        if states.ndim != 3 or states.shape[1] != config.horizon or states.shape[2] != config.state_dim:
            reasons.append(
                f"states shape 不匹配: actual={states.shape}, expected=(*, {config.horizon}, {config.state_dim})"
            )
        if actions.ndim != 2 or actions.shape[0] != states.shape[0] or actions.shape[1] != config.horizon:
            reasons.append(
                f"actions shape 不匹配: actual={actions.shape}, expected=({states.shape[0]}, {config.horizon})"
            )
        if rewards.ndim != 2 or rewards.shape[0] != states.shape[0] or rewards.shape[1] != config.horizon:
            reasons.append(
                f"rewards shape 不匹配: actual={rewards.shape}, expected=({states.shape[0]}, {config.horizon})"
            )

        for key, expected in expected_values.items():
            raw_value = data[key]
            if isinstance(raw_value, np.ndarray) and raw_value.shape == ():
                actual = raw_value.item()
            elif isinstance(raw_value, np.ndarray) and raw_value.size == 1:
                actual = raw_value.reshape(()).item()
            else:
                actual = raw_value
            if isinstance(actual, bytes):
                actual = actual.decode("utf-8")
            if isinstance(expected, float):
                if not np.isclose(float(actual), expected, atol=1e-12, rtol=0.0):
                    reasons.append(f"{key} 不匹配: actual={actual}, expected={expected}")
            else:
                if actual != expected:
                    reasons.append(f"{key} 不匹配: actual={actual}, expected={expected}")

    return len(reasons) == 0, reasons


def backup_incompatible_cache(traj_path: str, reasons: List[str]) -> str:
    """备份不兼容的旧轨迹缓存，避免被当前严格论文运行误复用。

    Args:
        traj_path: 原缓存文件路径
        reasons: 不兼容原因列表

    Returns:
        备份文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = traj_path.replace(".npz", f".incompatible_{timestamp}.npz")
    shutil.move(traj_path, backup_path)
    logger.warning(
        "检测到现有 trajectory cache 与当前严格论文设置不兼容，已备份到 %s。原因: %s",
        backup_path,
        reasons,
    )
    return backup_path


def load_data_and_env(config: Any, pair: str) -> Tuple[TradingEnv, TradingEnv, int]:
    """加载特征数据并初始化 TradingEnv。

    Args:
        config: 配置对象
        pair: 交易对

    Returns:
        (训练环境, DP环境, 训练数据行数)
    """
    logger.info("加载特征数据: data_dir=%s, pair=%s", config.data_dir, pair)
    pipeline = FeaturePipeline(
        config.data_dir, pair, cycle_features=config.cycle_features,
    )
    train_df, _, _ = pipeline.get_state_vector()
    train_prices_df, _, _ = pipeline.get_prices()

    train_states = train_df.to_numpy()
    prices = train_prices_df["close"].to_numpy()
    train_rows = int(train_states.shape[0])

    logger.info("训练集: states shape=%s, prices shape=%s", train_states.shape, prices.shape)
    log_training_data_scale(train_rows)

    available_starts = expected_num_available_starts(train_rows, config.horizon)
    if available_starts < config.num_trajectories:
        raise ValueError(
            "当前训练集不足以在严格论文滑窗协议下无放回采样 30k trajectories。"
            f" available_starts={available_starts}, required={config.num_trajectories}"
        )

    env = TradingEnv(
        states=train_states,
        prices=prices,
        pair=pair,
        horizon=config.horizon,
        states_dataframe=train_df,
        max_positions=config.max_positions,
        commission_rate=config.train_commission_rate,
    )

    # DP planner 用更高的费率筛选高利润轨迹
    dp_env = TradingEnv(
        states=train_states,
        prices=prices,
        pair=pair,
        horizon=config.horizon,
        states_dataframe=train_df,
        max_positions=config.max_positions,
        commission_rate=config.dp_commission_rate,
    )

    logger.info(
        "TradingEnv 初始化完成: num_horizons=%d, horizon=%d, max_position=%d, "
        "train_commission_rate=%.6f, dp_commission_rate=%.6f, available_starts=%d",
        env.num_horizons, config.horizon, env.m,
        env.commission_rate, dp_env.commission_rate, available_starts,
    )
    return env, dp_env, train_rows


def prepare_trajectory_dataset(
    config: Any, pair: str, dp_env: TradingEnv, train_rows: int,
) -> Tuple[TrajectoryDataset, str]:
    """检查缓存 / 生成 DP 示范轨迹，返回 (dataset, traj_path)。

    使用 dp_env（高费率环境）生成轨迹，筛选出高利润交易模式。

    Args:
        config: 配置对象
        pair: 交易对
        dp_env: DP 环境（高费率）
        train_rows: 训练数据行数

    Returns:
        (轨迹数据集, 缓存文件路径)
    """
    planner = DPPlanner(
        env=dp_env,
        gamma=config.discount_factor,
        result_dir=config.result_dir,
        train_batch_id=config.train_batch_id,
        sampling_seed=config.phase1_sampling_seed,
        sampling_mode=config.phase1_start_sampling_mode,
        stratified_ratio=config.phase1_stratified_ratio,
        importance_ratio=config.phase1_importance_ratio,
        sampling_num_strata=config.phase1_sampling_strata,
        importance_vol_weight=config.phase1_importance_vol_weight,
        importance_net_weight=config.phase1_importance_net_weight,
    )
    traj_path = DPPlanner.build_trajectory_cache_path(
        config.result_dir, pair, config.train_batch_id,
    )

    if os.path.exists(traj_path):
        cache_ok, cache_reasons = inspect_trajectory_cache(
            traj_path=traj_path, config=config, pair=pair, train_rows=train_rows,
        )
        if cache_ok:
            logger.info("发现与当前严格论文设置兼容的轨迹缓存，直接加载: %s", traj_path)
            return TrajectoryDataset.from_npz(traj_path, normalize=True), traj_path
        backup_incompatible_cache(traj_path, cache_reasons)

    logger.info("开始生成 DP 示范轨迹: num_trajectories=%d", config.num_trajectories)
    trajectories = planner.generate_trajectories(config.num_trajectories)
    logger.info("DP 轨迹生成完成，创建 Dataset")
    dataset = TrajectoryDataset(
        states=trajectories["states"],
        actions=trajectories["actions"],
        rewards=trajectories["rewards"],
        normalize=True,
    )
    return dataset, traj_path


def build_val_env(config: Any, pair: str) -> TradingEnv | None:
    """构建验证集环境（若验证集不足一个 horizon，则返回 None）。

    Args:
        config: 配置对象
        pair: 交易对

    Returns:
        验证环境或 None
    """
    val_pipeline = FeaturePipeline(
        config.data_dir, pair, cycle_features=config.cycle_features,
    )
    _, val_state_df, _ = val_pipeline.get_state_vector()
    _, val_prices_df, _ = val_pipeline.get_prices()
    if val_state_df is None or len(val_state_df) < config.horizon:
        return None
    if val_state_df.width != config.state_dim:
        raise ValueError(
            "验证集 state_dim 与当前配置不一致: "
            f"actual={val_state_df.width}, expected={config.state_dim}"
        )
    return TradingEnv(
        states=val_state_df.to_numpy(),
        prices=val_prices_df["close"].to_numpy(),
        pair=pair,
        horizon=config.horizon,
        states_dataframe=val_state_df,
        max_positions=config.max_positions,
        commission_rate=config.train_commission_rate,
    )