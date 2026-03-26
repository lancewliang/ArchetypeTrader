"""DP Planner — Algorithm 1: 单次交易约束动态规划

# 论文 Algorithm 1: Single-trade DP planner
# 在每个 horizon 内，使用反向填表 + 前向追踪生成最优示范轨迹。
# 约束: 每个 horizon 内最多一次 round-trip 交易（开仓一次 + 平仓一次）。
#
# 状态表: V[N+1, |A|, 2]  — 从时刻 t 开始、当前动作 a、约束标志 c 下的最优累积奖励
# 策略表: Π[N, |A|, 2]    — 每个 (t, a, c) 对应的最优下一步动作
# 约束标志: c ∈ {0, 1}
#   c=0: 尚未进行过交易（仍处于 flat 状态）
#   c=1: 已经进行过交易（已进入过非 flat 持仓）

需求: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 9.1
"""

import os
from typing import Dict, Tuple, Union

import numpy as np
import polars as pl

from src.env.trading_env import TradingEnv
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DPPlanner:
    """单次交易约束动态规划规划器。

    # 论文 Algorithm 1: Single-trade DP planner
    # 输入: 状态序列 S (长度 N), 价格序列 P (长度 N), 动作集 A = {0, 1, 2}
    # 输出: 最优示范轨迹 (s_demo, a_demo, r_demo)
    """

    # 动作集: {0: short, 1: flat, 2: long}
    ACTIONS = [0, 1, 2]
    NUM_ACTIONS = 3
    # flat 动作索引
    FLAT_ACTION = 1
    # 约束标志: c ∈ {0, 1}
    C = 2
    def __init__(self, env: TradingEnv, gamma: float = 0.99):
        """
        Args:
            env: MDP 交易环境实例，提供奖励计算和持仓映射。
            gamma: 折扣因子 γ，默认 0.99（与论文 Algorithm 1 一致）。
        """
        self.env = env
        self.pair = env.pair
        self.horizon = env.horizon
        self.m = env.m  # 最大持仓量
        self.gamma = gamma

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def plan(
        self, states: pl.DataFrame, prices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """对单个 horizon 执行动态规划，生成最优示范轨迹。

        # Algorithm 1, Step 1: 初始化
        # Algorithm 1, Step 2: 反向填表 (Backward pass)
        # Algorithm 1, Step 3: 前向追踪 (Forward pass)

        Args:
            states: 单个 horizon 的状态序列, polars DataFrame
            prices: 单个 horizon 的价格序列, shape (N,)
                    需要 N+1 个价格点来计算 N 步奖励，
                    但如果只有 N 个价格，最后一步价差为 0。

        Returns:
            s_demo: 状态序列 (N, state_dim)
            a_demo: 动作序列 (N,) 值域 {0, 1, 2}
            r_demo: 奖励序列 (N,)
        """
        N = len(states)
        A = self.NUM_ACTIONS

        # ============================================================
        # Algorithm 1, Step 1: 初始化
        # V[N, a, c] = 0 for all a, c
        # ============================================================

        V = np.zeros((N + 1, A, self.C), dtype=np.float64)
        Pi = np.full((N, A, self.C), self.FLAT_ACTION, dtype=np.int32)

        # ============================================================
        # Algorithm 1, Step 2: 反向填表 (Backward pass)
        # 从 t = N-1 反向到 t = 0
        # ============================================================
        for t in range(N - 1, -1, -1):
            # 当前价格和下一步价格（用于计算奖励）
            p_t = prices[t]
            p_next = prices[t + 1] if (t + 1) < len(prices) else prices[t]

            for a_prev in range(A):
                # 当前持仓（由上一步动作决定）
                prev_position = TradingEnv.POSITION_MAP[a_prev] * self.m

                for c in range(self.C):
                    best_val = -np.inf
                    best_action = self.FLAT_ACTION

                    for a_next in range(A):
                        # 检查约束: 单次交易约束
                        # c=0 表示尚未交易过
                        # c=1 表示已经交易过（已进入过非 flat 持仓）
                        next_position = TradingEnv.POSITION_MAP[a_next] * self.m

                        # 计算约束转移
                        c_next = self._compute_next_constraint(
                            c, a_prev, a_next
                        )
                        if c_next < 0:
                            # 违反约束，跳过
                            continue

                        # 计算即时奖励
                        # r_step_t = new_position * (p_{t+1} - p_t) - execution_cost
                        execution_cost = self.env.compute_execution_cost(
                            a_next, prev_position, p_t, states.row(t, named=True)
                        )
                        reward = next_position * (p_next - p_t) - execution_cost

                        # Bellman 方程: V[t, a_prev, c] = max_a { r(t, a_prev→a) + γ × V[t+1, a, c'] }
                        val = reward + self.gamma * V[t + 1, a_next, c_next]

                        if val > best_val:
                            best_val = val
                            best_action = a_next

                    V[t, a_prev, c] = best_val
                    Pi[t, a_prev, c] = best_action

        # ============================================================
        # Algorithm 1, Step 3: 前向追踪 (Forward pass)
        # 从 t=0 开始，初始状态为 flat (a=1, c=0)
        # ============================================================
        a_demo = np.empty(N, dtype=np.int32)
        r_demo = np.empty(N, dtype=np.float64)

        # 初始状态: flat, 未交易
        current_action_idx = self.FLAT_ACTION  # flat
        current_c = 0  # 未交易
        current_position = 0  # flat 持仓

        # Algorithm 1, Step 9-12: 前向追踪 t = 0 到 N-2
        for t in range(N - 1):
            # 选择最优动作
            next_action = Pi[t, current_action_idx, current_c]
            next_position = TradingEnv.POSITION_MAP[next_action] * self.m

            # 计算即时奖励
            p_t = prices[t]
            p_next = prices[t + 1] if (t + 1) < len(prices) else prices[t]
            execution_cost = self.env.compute_execution_cost(
                next_action, current_position, p_t, states.row(t, named=True)
            )
            reward = next_position * (p_next - p_t) - execution_cost

            a_demo[t] = next_action
            r_demo[t] = reward

            # 更新约束标志
            current_c = self._compute_next_constraint(
                current_c, current_action_idx, next_action
            )
            current_action_idx = next_action
            current_position = next_position

        # Algorithm 1, Step 13: â_{N-1} ← â_{N-2}
        if N >= 2:
            a_demo[N - 1] = a_demo[N - 2]
        else:
            # N=1 的边界情况：只有一步，使用 Pi 表
            a_demo[0] = Pi[0, self.FLAT_ACTION, 0]

        # 计算最后一步的奖励
        last_action = int(a_demo[N - 1])
        last_position = TradingEnv.POSITION_MAP[last_action] * self.m
        p_t = prices[N - 1]
        p_next = prices[N] if N < len(prices) else prices[N - 1]
        execution_cost = self.env.compute_execution_cost(
            last_action, current_position, p_t, states.row(N - 1, named=True)
        )
        r_demo[N - 1] = last_position * (p_next - p_t) - execution_cost

        s_demo = states.to_numpy().copy()
        return s_demo, a_demo, r_demo

    def generate_trajectories(
        self, num_trajectories: int = 30000
    ) -> Dict[str, np.ndarray]:
        """批量生成示范轨迹。

        遍历环境中所有可用 horizon，对每个 horizon 执行 DP 规划。
        如果 horizon 数量不足 num_trajectories，则循环复用。

        Args:
            num_trajectories: 目标轨迹数量，默认 30000。

        Returns:
            字典包含:
                'states':  (num_trajectories, h, state_dim)
                'actions': (num_trajectories, h)
                'rewards': (num_trajectories, h)
        """
        num_horizons = self.env.num_horizons
        if num_horizons == 0:
            logger.warning(
                "环境中无可用 horizon，生成全 flat 轨迹"
            )
            return self._generate_all_flat(num_trajectories)

        logger.info(
            "开始生成 DP 示范轨迹: pair=%s, num_horizons=%d, target=%d",
            self.pair,
            num_horizons,
            num_trajectories,
        )

        # 对每个 horizon 执行一次 DP 规划
        if self.env.states_dataframe is None:
            raise ValueError(
                f"DPPlanner.generate_trajectories() 需要 env.states_dataframe，"
                f"但当前为 None。请在创建 TradingEnv 时传入 states_dataframe 参数。"
            )

        horizon_results = []
        for h_idx in range(num_horizons):
            start = h_idx * self.horizon
            end = start + self.horizon
            h_states = self.env.states_dataframe[start:end]
            # 价格需要多取一个点用于最后一步奖励计算
            price_end = min(end + 1, len(self.env.prices))
            h_prices = self.env.prices[start:price_end]
            
            s_demo, a_demo, r_demo = self.plan(h_states, h_prices)

            if h_idx == 0:
                logger.info(
                    "Horizon 0 shapes: h_states=%s, h_prices=%s, s_demo=%s, a_demo=%s, r_demo=%s",
                    h_states.shape,
                    h_prices.shape,
                    s_demo.shape,
                    a_demo.shape,
                    r_demo.shape,
                )

            # 检查是否为全 flat 轨迹（无有效交易路径）
            if np.all(a_demo == self.FLAT_ACTION):
                logger.warning(
                    "Horizon %d: 无有效交易路径，输出全 flat 轨迹",
                    h_idx,
                )

            horizon_results.append((s_demo, a_demo, r_demo))

        # 循环复用 horizon 结果以达到目标数量
        all_states = []
        all_actions = []
        all_rewards = []

        for i in range(num_trajectories):
            idx = i % len(horizon_results)
            s, a, r = horizon_results[idx]
            all_states.append(s)
            all_actions.append(a)
            all_rewards.append(r)

        result = {
            "states": np.array(all_states, dtype=np.float32),
            "actions": np.array(all_actions, dtype=np.int32),
            "rewards": np.array(all_rewards, dtype=np.float32),
        }

        logger.info(
            "DP 轨迹生成完成: states=%s, actions=%s, rewards=%s",
            result["states"].shape,
            result["actions"].shape,
            result["rewards"].shape,
        )

        # 保存轨迹到文件
        self._save_trajectories(result)

        return result

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _compute_next_constraint(
        self, c: int, a_prev: int, a_next: int
    ) -> int:
        """计算约束标志的转移。

        单次交易约束逻辑（每个 horizon 最多一次 round-trip）:
        - c=0 (未交易): 可以保持 flat，或者开仓（flat → non-flat）→ c 变为 1
        - c=1 (已交易/持仓中): 可以保持同一持仓方向，或者平仓回 flat，
                               但不能切换持仓方向（short↔long），
                               且平仓后不能再开新仓

        允许的完整路径:
          flat → flat → ... (从不交易)
          flat → long → long → ... → long (开仓后一直持有)
          flat → long → long → ... → flat → flat → ... (一次 round-trip)
          flat → short → short → ... → flat → flat → ... (一次 round-trip)

        Args:
            c: 当前约束标志 (0 或 1)
            a_prev: 上一步动作（代表当前持仓方向）
            a_next: 当前动作（代表目标持仓方向）

        Returns:
            新的约束标志 (0 或 1)，-1 表示违反约束。
        """
        prev_is_flat = (a_prev == self.FLAT_ACTION)
        next_is_flat = (a_next == self.FLAT_ACTION)

        if c == 0:
            # 尚未交易
            if next_is_flat:
                # 保持 flat，约束不变
                return 0
            else:
                # 开仓: flat → non-flat，标记为已交易
                return 1
        else:
            # c == 1: 已经进行过交易
            if prev_is_flat and not next_is_flat:
                # 已经平仓回 flat 后又想开新仓 → 违反单次交易约束
                return -1
            if not prev_is_flat and not next_is_flat and a_prev != a_next:
                # 直接切换持仓方向 (short → long 或 long → short)
                # 这相当于两次交易，违反约束
                return -1
            # 允许: 保持同一持仓、平仓回 flat
            return 1

    def _generate_all_flat(
        self, num_trajectories: int
    ) -> Dict[str, np.ndarray]:
        """生成全 flat 轨迹（无有效交易路径时的回退方案）。

        Args:
            num_trajectories: 轨迹数量

        Returns:
            全 flat 轨迹字典
        """
        logger.warning(
            "生成 %d 条全 flat 轨迹 (pair=%s)",
            num_trajectories,
            self.pair,
        )
        state_dim = self.env.state_dim
        h = self.horizon

        states = np.zeros(
            (num_trajectories, h, state_dim), dtype=np.float32
        )
        actions = np.full(
            (num_trajectories, h), self.FLAT_ACTION, dtype=np.int32
        )
        rewards = np.zeros((num_trajectories, h), dtype=np.float32)

        return {"states": states, "actions": actions, "rewards": rewards}

    def _save_trajectories(self, trajectories: Dict[str, np.ndarray]) -> None:
        """保存轨迹到 result/dp_trajectories/{pair}_trajectories.npz。

        Args:
            trajectories: 轨迹字典
        """
        save_dir = os.path.join("result", "dp_trajectories")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.pair}_trajectories.npz")

        np.savez(
            save_path,
            states=trajectories["states"],
            actions=trajectories["actions"],
            rewards=trajectories["rewards"],
        )
        logger.info("轨迹已保存到 %s", save_path)
