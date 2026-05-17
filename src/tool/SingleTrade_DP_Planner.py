"""Single-trade DP planner 的接口骨架。"""

from __future__ import annotations

import numpy as np

from ..model.data_types import DemonstrationTrajectory, HorizonDataset, TrajectoryDataset


class SingleTrade_DP_Planner:
    """为固定 horizon 生成 single-trade demonstration trajectories。

    该类对应论文中的 Single-trade DP planner。
    它的目标是在每个长度为 ``h`` 的 horizon 内，根据价格序列寻找一条
    最优或高质量的 teacher action 序列，并把状态、动作和奖励组合成：

    ``tau = (s_demo, a_demo, r_demo)``

    其中:
        ``s_demo`` 的 shape 为 ``[h, feature_dim]``。
        ``a_demo`` 的 shape 为 ``[h]``，动作取值为 ``{0, 1, 2}``。
        ``r_demo`` 的 shape 为 ``[h]``，表示逐步 reward。

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
    ) -> None:
        """初始化 Single-trade DP planner。

        参数:
            horizon: 每条 demonstration trajectory 的时间窗口长度 ``h``。
            action_set: 可选动作集合，默认 ``(0, 1, 2)``，分别表示 short、flat、long。
            initial_action: 每个 horizon 开始时的初始仓位动作，默认 ``1`` 表示 flat。
            gamma: DP 规划中的折扣系数，用于累计未来 reward。

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

        self.horizon = horizon
        self.action_set = tuple(action_set)
        self.initial_action = initial_action
        self.gamma = gamma
        self._action_to_position = {0: -1.0, 1: 0.0, 2: 1.0}
   
    def build_trajectory(
        self,
        states: np.ndarray,
        prices: np.ndarray,
    ) -> DemonstrationTrajectory:
        """为单个 horizon 生成 demonstration trajectory。

        参数:
            states: 单个 horizon 的市场状态矩阵，shape 为 ``[h, feature_dim]``。
            prices: 单个 horizon 的价格序列，shape 为 ``[h]``。

        输出:
            返回 ``DemonstrationTrajectory``，即 ``tau = (s_demo, a_demo, r_demo)``。
            ``s_demo`` 的 shape 为 ``[h, feature_dim]``。
            ``a_demo`` 的 shape 为 ``[h]``。
            ``r_demo`` 的 shape 为 ``[h]``。

        方法作用:
            先调用 ``plan`` 生成 teacher actions，再调用 ``compute_rewards`` 生成
            step rewards，最后把 ``states``、``actions``、``rewards`` 组合成论文中的
            demonstration tuple。

        为什么:
            VQ encoder-decoder 的训练样本单位是完整的 ``tau``，
            而不是孤立的动作序列。该方法把 DP teacher 输出整理成模型训练契约。
        """
        states = np.asarray(states, dtype=np.float32)
        prices = np.asarray(prices, dtype=np.float32).reshape(-1)
        if states.ndim != 2:
            raise ValueError("states must have shape [h, feature_dim]")
        if prices.ndim != 1:
            raise ValueError("prices must have shape [h]")
        if states.shape[0] != prices.shape[0]:
            raise ValueError("states and prices must share horizon length")
        if states.shape[0] != self.horizon:
            raise ValueError(
                f"expected horizon={self.horizon}, got {states.shape[0]}"
            )

        actions = self.plan(prices)
        rewards = self.compute_rewards(prices, actions)
        return states, actions, rewards

    def build_trajectory_dataset(
        self,
        horizon_dataset: HorizonDataset,
    ) -> TrajectoryDataset:
        """批量生成 demonstration trajectory 数据集 ``D``。

        参数:
            horizon_dataset: ``HorizonBuilder`` 的输出，
                即 ``(states, prices, depthprices)``。
                ``states`` 的 shape 为 ``[n, h, feature_dim]``。
                ``prices`` 的 shape 为 ``[n, h, 1]``，来自 feature 文件的 ``close`` 列。

        输出:
            返回 ``TrajectoryDataset``，即 ``D = [tau_0, tau_1, ..., tau_{n-1}]``。
            每个 ``tau_i`` 都是 ``(s_demo, a_demo, r_demo)``。

        方法作用:
            从 ``horizon_dataset`` 中取出每个 horizon 的 ``states`` 和 ``prices``，
            对每个 sampled horizon 调用 ``build_trajectory``，
            批量生成 Phase I 训练所需的 demonstration trajectories。

        为什么:
            论文中的训练数据集定义为 ``D = {tau_i}_{i=0}^{n-1}``。
            该方法把单条 trajectory 生成逻辑扩展到完整训练集。
        """
        states_batch, prices_batch, _depthprices_batch = horizon_dataset
        states_batch = np.asarray(states_batch, dtype=np.float32)
        prices_batch = np.asarray(prices_batch, dtype=np.float32)
        if states_batch.ndim != 3:
            raise ValueError("states must have shape [n, h, feature_dim]")
        if prices_batch.ndim != 3 or prices_batch.shape[-1] != 1:
            raise ValueError("prices must have shape [n, h, 1]")
        if states_batch.shape[:2] != prices_batch.shape[:2]:
            raise ValueError("states and prices must share [n, h]")
        if states_batch.shape[1] != self.horizon:
            raise ValueError(
                f"expected horizon={self.horizon}, got {states_batch.shape[1]}"
            )

        return [
            self.build_trajectory(states_batch[index], prices_batch[index, :, 0])
            for index in range(states_batch.shape[0])
        ]

    def plan(self, prices: np.ndarray) -> np.ndarray:
        """按 Algorithm 1 的单次换仓约束生成 teacher action 序列。"""

        prices = np.asarray(prices, dtype=np.float32).reshape(-1)
        horizon = prices.shape[0]
        if horizon != self.horizon:
            raise ValueError(f"expected horizon={self.horizon}, got {horizon}")

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
                            action=next_action,
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
                changed + int(self.action_set[current_idx] != self.action_set[next_idx]),
            )
            current_idx = next_idx
        actions[-1] = actions[-2]
        return actions

    def compute_rewards(self, prices: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """根据 teacher actions 计算逐步持仓收益。"""

        prices = np.asarray(prices, dtype=np.float32).reshape(-1)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if prices.shape[0] != actions.shape[0]:
            raise ValueError("prices and actions must have the same length")

        rewards = np.zeros_like(prices, dtype=np.float32)
        for t in range(prices.shape[0] - 1):
            rewards[t] = self._step_reward(prices[t], prices[t + 1], int(actions[t]))
        rewards[-1] = 0.0
        return rewards

    def _step_reward(
        self,
        price_now: float,
        price_next: float,
        action: int,
    ) -> float:
        position = self._action_to_position.get(action)
        if position is None:
            raise ValueError(f"unsupported action: {action}")
        return float(position * (price_next - price_now))
 
