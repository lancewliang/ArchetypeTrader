"""Single-trade DP planner 的接口骨架。"""

from __future__ import annotations

import numpy as np

from ..model.data_types import DemonstrationTrajectory, HorizonDataset, TrajectoryDataset
from ..utils.trade_execution import (
    ActionExecutionCalculator,
    LOB_DEPTH_WIDTH,
)


class SingleTrade_DP_Planner:
    """为固定 horizon 生成 single-trade demonstration trajectories。

    该类对应论文中的 Single-trade DP planner。
    它的目标是在每个长度为 ``h`` 的 horizon 内，根据价格序列寻找一条
    最优或高质量的 teacher action 序列，并把状态、动作和奖励组合成：

    ``tau = (s_demo, relative_s_demo, trend_s_demo, a_demo, r_demo, sample_id)``

    其中:
        ``s_demo`` 的 shape 为 ``[h, feature_dim]``。
        ``a_demo`` 的 shape 为 ``[h]``，动作取值为 ``{0, 1, 2}``。
        ``r_demo`` 的 shape 为 ``[h]``，表示逐步 reward。
        ``sample_id`` 是 split 内稳定 horizon 样本编号。

    为什么需要这个类:
        Phase I 的 VQ encoder-decoder 需要 demonstration trajectories
        作为训练数据。DP teacher 只在离线训练数据生成阶段使用，避免在训练、
        验证、测试或推理阶段重新调用未来信息。
    """

    def __init__(
        self,
        horizon: int = 72,
        action_set: tuple[int, ...] = (0, 1, 2),
        initial_action: int = 1,
        gamma: float = 1.0,
        fee_rate: float = 0.0004,
    ) -> None:
        """初始化 Single-trade DP planner。

        参数:
            horizon: 每条 demonstration trajectory 的时间窗口长度 ``h``。
            action_set: 可选动作集合，默认 ``(0, 1, 2)``，分别表示 short、flat、long。
            initial_action: 每个 horizon 开始时的初始仓位动作，默认 ``1`` 表示 flat。
            gamma: DP 规划中的折扣系数，用于累计未来 reward。
            fee_rate: 单边交易费率，按 ``abs(delta_position) * mark_price`` 计费。

        输出:
            无返回值。

        方法作用:
            记录 DP planner 的基本配置，确保后续
            ``build_trajectory`` 使用一致的 horizon、动作空间和初始仓位假设。

        为什么:
            demonstration trajectories 必须由同一套交易语义生成，否则 Phase I
            学到的 archetype 会混合不一致的动作和 reward 定义。
        """
        if horizon <= 1:
            raise ValueError("horizon must be greater than 1")
        if initial_action not in action_set:
            raise ValueError("initial_action must be included in action_set")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if fee_rate < 0:
            raise ValueError("fee_rate must be non-negative")

        self.horizon = horizon
        self.action_set = tuple(action_set)
        self.initial_action = initial_action
        self.gamma = gamma
        self.fee_rate = float(fee_rate)
        self._action_to_position = {0: -1.0, 1: 0.0, 2: 1.0}

    def build_trajectory(
        self,
        states: np.ndarray,
        relative_states: np.ndarray,
        trend_states: np.ndarray,
        prices: np.ndarray,
        depthprices: np.ndarray,
        sample_id: int,
    ) -> DemonstrationTrajectory:
        """为单个 horizon 生成 demonstration trajectory。

        参数:
            states: 单个 horizon 的市场状态矩阵，shape 为 ``[h, feature_dim]``。
            relative_states: 单个 horizon 的相对状态矩阵，
                shape 为 ``[h, relative_feature_dim]``。
            trend_states: 单个 horizon 的趋势状态矩阵，
                shape 为 ``[h, trend_feature_dim]``。
            prices: 单个 horizon 的价格序列，shape 为 ``[h]``。
            depthprices: 单个 horizon 的 LOB 深度行情，shape 为 ``[h, 20]``。
            sample_id: 当前 horizon trajectory 在 split 内的稳定样本编号。

        输出:
            返回 ``DemonstrationTrajectory``，即
            ``tau = (
                s_demo, relative_s_demo, trend_s_demo, a_demo, r_demo, sample_id
            )``。
            ``s_demo`` 的 shape 为 ``[h, feature_dim]``。
            ``relative_s_demo`` 的 shape 为 ``[h, relative_feature_dim]``。
            ``trend_s_demo`` 的 shape 为 ``[h, trend_feature_dim]``。
            ``a_demo`` 的 shape 为 ``[h]``。
            ``r_demo`` 的 shape 为 ``[h]``。
            ``sample_id`` 是 scalar int。

        方法作用:
            先调用 ``plan`` 生成 teacher actions，再调用 ``compute_rewards`` 生成
            step rewards，最后把 ``states``、``actions``、``rewards`` 组合成论文中的
            demonstration tuple。

        为什么:
            VQ encoder-decoder 的训练样本单位是完整的 ``tau``，
            而不是孤立的动作序列。该方法把 DP teacher 输出整理成模型训练契约。
        """
        states = np.asarray(states, dtype=np.float32)
        relative_states = np.asarray(relative_states, dtype=np.float32)
        trend_states = np.asarray(trend_states, dtype=np.float32)
        prices = np.asarray(prices, dtype=np.float32).reshape(-1)
        depthprices = np.asarray(depthprices, dtype=np.float32)
        if states.ndim != 2:
            raise ValueError("states must have shape [h, feature_dim]")
        if relative_states.ndim != 2:
            raise ValueError(
                "relative_states must have shape [h, relative_feature_dim]"
            )
        if trend_states.ndim != 2:
            raise ValueError("trend_states must have shape [h, trend_feature_dim]")
        if prices.ndim != 1:
            raise ValueError("prices must have shape [h]")
        if depthprices.ndim != 2 or depthprices.shape[1] != LOB_DEPTH_WIDTH:
            raise ValueError(f"depthprices must have shape [h, {LOB_DEPTH_WIDTH}]")
        if states.shape[0] != prices.shape[0]:
            raise ValueError("states and prices must share horizon length")
        if relative_states.shape[0] != prices.shape[0]:
            raise ValueError("relative_states and prices must share horizon length")
        if trend_states.shape[0] != prices.shape[0]:
            raise ValueError("trend_states and prices must share horizon length")
        if states.shape[0] != depthprices.shape[0]:
            raise ValueError("states and depthprices must share horizon length")
        if states.shape[0] != self.horizon:
            raise ValueError(
                f"expected horizon={self.horizon}, got {states.shape[0]}"
            )

        actions = self.plan(prices, depthprices)
        rewards = self.compute_rewards(prices, actions, depthprices)
        return states, relative_states, trend_states, actions, rewards, int(sample_id)

    def build_trajectory_dataset(
        self,
        horizon_dataset: HorizonDataset,
    ) -> TrajectoryDataset:
        """批量生成 demonstration trajectory 数据集 ``D``。

        参数:
            horizon_dataset: ``HorizonBuilder`` 的输出，
                即 ``(states, relative_states, trend_states, prices, depthprices)``。
                ``states`` 的 shape 为 ``[n, h, feature_dim]``。
                ``relative_states`` 的 shape 为 ``[n, h, relative_feature_dim]``。
                ``trend_states`` 的 shape 为 ``[n, h, trend_feature_dim]``。
                ``prices`` 的 shape 为 ``[n, h, 1]``，来自 feature 文件的 ``close`` 列。
                ``depthprices`` 的 shape 为 ``[n, h, 20]``，来自 states 中的 LOB
                深度行情。

        输出:
            返回 ``TrajectoryDataset``，即 ``D = [tau_0, tau_1, ..., tau_{n-1}]``。
            每个 ``tau_i`` 都是
            ``(s_demo, relative_s_demo, trend_s_demo, a_demo, r_demo, sample_id)``。

        方法作用:
            从 ``horizon_dataset`` 中取出每个 horizon 的 ``states``、``prices`` 和
            ``depthprices``，对每个 sampled horizon 调用 ``build_trajectory``，
            批量生成 Phase I 训练所需的 demonstration trajectories。

        为什么:
            论文中的训练数据集定义为 ``D = {tau_i}_{i=0}^{n-1}``。
            该方法把单条 trajectory 生成逻辑扩展到完整训练集。
        """
        (
                states_batch,
                relative_states_batch,
                trend_states_batch,
                prices_batch,
                depthprices_batch,
            ) = horizon_dataset
        states_batch = np.asarray(states_batch, dtype=np.float32)
        relative_states_batch = np.asarray(relative_states_batch, dtype=np.float32)
        trend_states_batch = np.asarray(trend_states_batch, dtype=np.float32)
        prices_batch = np.asarray(prices_batch, dtype=np.float32)
        depthprices_batch = np.asarray(depthprices_batch, dtype=np.float32)
        if states_batch.ndim != 3:
            raise ValueError("states must have shape [n, h, feature_dim]")
        if relative_states_batch.ndim != 3:
            raise ValueError(
                "relative_states must have shape [n, h, relative_feature_dim]"
            )
        if trend_states_batch.ndim != 3:
            raise ValueError("trend_states must have shape [n, h, trend_feature_dim]")
        if prices_batch.ndim != 3 or prices_batch.shape[-1] != 1:
            raise ValueError("prices must have shape [n, h, 1]")
        if (
            depthprices_batch.ndim != 3
            or depthprices_batch.shape[-1] != LOB_DEPTH_WIDTH
        ):
            raise ValueError(f"depthprices must have shape [n, h, {LOB_DEPTH_WIDTH}]")
        if states_batch.shape[:2] != prices_batch.shape[:2]:
            raise ValueError("states and prices must share [n, h]")
        if relative_states_batch.shape[:2] != prices_batch.shape[:2]:
            raise ValueError("relative_states and prices must share [n, h]")
        if trend_states_batch.shape[:2] != prices_batch.shape[:2]:
            raise ValueError("trend_states and prices must share [n, h]")
        if states_batch.shape[:2] != depthprices_batch.shape[:2]:
            raise ValueError("states and depthprices must share [n, h]")
        if states_batch.shape[1] != self.horizon:
            raise ValueError(
                f"expected horizon={self.horizon}, got {states_batch.shape[1]}"
            )

        return [
            self.build_trajectory(
                states_batch[index],
                relative_states_batch[index],
                trend_states_batch[index],
                prices_batch[index, :, 0],
                depthprices_batch[index],
                sample_id=index,
            )
            for index in range(states_batch.shape[0])
        ]

    def plan(self, prices: np.ndarray, depthprices: np.ndarray) -> np.ndarray:
        """按 Algorithm 1 的单次换仓约束生成扣除执行成本后的 teacher action 序列。"""

        prices = np.asarray(prices, dtype=np.float32).reshape(-1)
        depthprices = np.asarray(depthprices, dtype=np.float32)
        horizon = prices.shape[0]
        if horizon != self.horizon:
            raise ValueError(f"expected horizon={self.horizon}, got {horizon}")
        if depthprices.shape != (horizon, LOB_DEPTH_WIDTH):
            raise ValueError(
                f"depthprices must have shape [{horizon}, {LOB_DEPTH_WIDTH}]"
            )

        num_actions = len(self.action_set)
        action_to_index = {
            action: index for index, action in enumerate(self.action_set)
        }
        initial_index = action_to_index[self.initial_action]
        values = np.zeros((horizon + 1, num_actions, 2), dtype=np.float32)
        policy = np.zeros((horizon, num_actions, 2), dtype=np.int64)

        for t in range(horizon - 2, -1, -1):
            for current_idx, current_action in enumerate(self.action_set):
                for changed in (0, 1):
                    best_value = -np.inf
                    best_next_idx = current_idx
                    for next_idx, next_action in enumerate(self.action_set):
                        next_changed = changed + int(current_action != next_action)
                        if next_changed > 1:
                            continue
                        reward = self._step_reward(
                            price_now=prices[t],
                            price_next=prices[t + 1],
                            current_action=current_action,
                            next_action=next_action,
                            depthprice=depthprices[t],
                        )
                        candidate = (
                            reward
                            + self.gamma
                            * values[t + 1, next_idx, min(next_changed, 1)]
                        )
                        if candidate > best_value:
                            best_value = candidate
                            best_next_idx = next_idx
                    values[t, current_idx, changed] = best_value
                    policy[t, current_idx, changed] = best_next_idx

        actions = np.empty(horizon, dtype=np.int64)
        current_idx = initial_index
        changed = 0
        for t in range(horizon - 1):
            next_idx = int(policy[t, current_idx, changed])
            actions[t] = self.action_set[next_idx]
            changed = min(
                1,
                changed
                + int(self.action_set[current_idx] != self.action_set[next_idx]),
            )
            current_idx = next_idx
        actions[-1] = actions[-2]
        return actions

    def compute_rewards(
        self,
        prices: np.ndarray,
        actions: np.ndarray,
        depthprices: np.ndarray,
    ) -> np.ndarray:
        """根据 teacher actions 计算扣除手续费和 LOB 滑点后的逐步持仓收益。"""

        prices = np.asarray(prices, dtype=np.float32).reshape(-1)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        depthprices = np.asarray(depthprices, dtype=np.float32)
        if prices.shape[0] != actions.shape[0]:
            raise ValueError("prices and actions must have the same length")
        if depthprices.shape != (prices.shape[0], LOB_DEPTH_WIDTH):
            raise ValueError(
                f"depthprices must have shape [{prices.shape[0]}, {LOB_DEPTH_WIDTH}]"
            )

        rewards = np.zeros_like(prices, dtype=np.float32)
        current_action = self.initial_action
        for t in range(prices.shape[0] - 1):
            next_action = int(actions[t])
            rewards[t] = self._step_reward(
                prices[t],
                prices[t + 1],
                current_action=current_action,
                next_action=next_action,
                depthprice=depthprices[t],
            )
            current_action = next_action
        rewards[-1] = 0.0
        return rewards

    def _step_reward(
        self,
        price_now: float,
        price_next: float,
        current_action: int,
        next_action: int,
        depthprice: np.ndarray,
    ) -> float:
        current_position = self._position_for_action(current_action)
        next_position = self._position_for_action(next_action)
        delta_position = next_position - current_position
        gross_reward = next_position * (price_next - price_now)
        fee_cost = abs(delta_position) * float(price_now) * self.fee_rate
        slippage_cost = ActionExecutionCalculator.compute_lob_slippage_from_depthprice(
            delta_position=delta_position,
            depthprice=depthprice,
            mark_price=float(price_now),
        )
        return float(gross_reward - fee_cost - slippage_cost)

    def _position_for_action(self, action: int) -> float:
        position = self._action_to_position.get(action)
        if position is None:
            raise ValueError(f"unsupported action: {action}")
        return position
