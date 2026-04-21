"""Phase II 数据加载模块

功能说明:
    负责加载特征数据、价格数据、DP 示范轨迹，
    并创建训练和验证环境。
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from src.data.feature_pipeline import FeaturePipeline
from src.env.trading_env import TradingEnv
from src.utils.logger import get_logger
from src.utils.normalizer import StateNormalizer

logger = get_logger(__name__)


def load_feature_data(
    config: Any,
    pair: str,
    normalizer: StateNormalizer | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any]:
    """加载特征数据和价格数据。

    Args:
        config: 配置对象
        pair: 交易对
        normalizer: 归一化器（如果提供，将对状态进行归一化）

    Returns:
        train_states, val_states, train_prices, val_prices, train_df, val_df
    """
    logger.info("加载特征数据: data_dir=%s, pair=%s", config.data_dir, pair)
    pipeline = FeaturePipeline(
        config.data_dir, pair, cycle_features=config.cycle_features,
    )
    train_df, val_df, _ = pipeline.get_state_vector()
    train_prices_df, val_prices_df, _ = pipeline.get_prices()

    train_states = train_df.to_numpy()
    val_states = val_df.to_numpy()
    train_prices = train_prices_df["close"].to_numpy()
    val_prices = val_prices_df["close"].to_numpy()

    # 归一化 states（与 Phase 1 训练一致）
    if normalizer is not None:
        train_states = normalizer.normalize_states(train_states)
        val_states = normalizer.normalize_states(val_states)

    logger.info(
        "训练集: states shape=%s, 验证集: states shape=%s",
        train_states.shape,
        val_states.shape,
    )

    return train_states, val_states, train_prices, val_prices, train_df, val_df


def create_environments(
    config: Any,
    pair: str,
    train_states: np.ndarray,
    val_states: np.ndarray,
    train_prices: np.ndarray,
    val_prices: np.ndarray,
    train_df: Any,
    val_df: Any,
) -> tuple[TradingEnv, TradingEnv]:
    """创建训练和验证环境。

    Args:
        config: 配置对象
        pair: 交易对
        train_states: 训练集状态
        val_states: 验证集状态
        train_prices: 训练集价格
        val_prices: 验证集价格
        train_df: 训练集 DataFrame
        val_df: 验证集 DataFrame

    Returns:
        train_env, val_env
    """
    train_env = TradingEnv(
        states=train_states,
        prices=train_prices,
        pair=pair,
        horizon=config.horizon,
        states_dataframe=train_df,
        max_positions=config.max_positions,
        commission_rate=config.train_commission_rate,
    )
    val_env = TradingEnv(
        states=val_states,
        prices=val_prices,
        pair=pair,
        horizon=config.horizon,
        states_dataframe=val_df,
        max_positions=config.max_positions,
        commission_rate=config.train_commission_rate,
    )
    logger.info(
        "TradingEnv 初始化完成: train_horizons=%d, val_horizons=%d",
        train_env.num_horizons,
        val_env.num_horizons,
    )
    return train_env, val_env


def load_demo_trajectories(
    config: Any,
    pair: str,
    normalizer: StateNormalizer | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载 DP 示范轨迹（用于 Eq.5 的 ground-truth archetype label）。

    DP 轨迹文件由 Phase I 的 DPPlanner.generate_trajectories() 生成，
    前 num_horizons 条与训练环境 horizon 索引 1:1 对齐。

    Args:
        config: 配置对象
        pair: 交易对
        normalizer: 归一化器（如果提供，将对状态和奖励进行归一化）

    Returns:
        demo_states, demo_actions, demo_rewards
    """
    traj_path = os.path.join(
        config.get_stage_result_dir(pair, "dp_trajectories"), "trajectories.npz",
    )
    if not os.path.exists(traj_path):
        raise FileNotFoundError(
            f"DP 轨迹文件不存在: {traj_path}\n"
            f"请先运行 Phase I 训练: python scripts/train_phase1.py --pair {pair}"
        )

    demo_data = np.load(traj_path)
    demo_states = demo_data["states"]    # (N, h, state_dim)
    demo_actions = demo_data["actions"]  # (N, h)
    demo_rewards = demo_data["rewards"]  # (N, h)

    # 归一化 demo 数据（与 Phase 1 训练时一致）
    if normalizer is not None:
        demo_states = normalizer.normalize_states(demo_states)
        demo_rewards = normalizer.normalize_rewards(demo_rewards)

    logger.info(
        "DP 示范轨迹加载完成: %d 条, horizon=%d",
        demo_states.shape[0],
        demo_states.shape[1],
    )

    return demo_states, demo_actions, demo_rewards
