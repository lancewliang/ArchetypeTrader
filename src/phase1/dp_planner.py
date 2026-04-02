"""DP Planner — Algorithm 1: 单次交易约束动态规划

# 论文 Algorithm 1: Single-trade DP planner
# 在每个 horizon 内，使用反向填表 + 前向追踪生成最优示范轨迹。
# 约束: 每个 horizon 内最多一次 action change（严格匹配论文中的 c + 1[i != j] <= 1）。
#
# 状态表: V[N+1, |A|, 2]  — 从时刻 t 开始、当前动作 a、约束标志 c 下的最优累积奖励
# 策略表: Π[N, |A|, 2]    — 每个 (t, a, c) 对应的最优下一步动作
# 约束标志: c ∈ {0, 1}
#   c=0: 尚未发生过动作变化
#   c=1: 已经发生过一次动作变化，后续不能再变化

需求: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 9.1
"""

import os
from typing import Any, Dict, Tuple

import numpy as np
import polars as pl
from tqdm import tqdm

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

    def __init__(
        self,
        env: TradingEnv,
        gamma: float = 0.99,
        result_dir: str = "result",
        sampling_seed: int = 42,
    ):
        """
        Args:
            env: MDP 交易环境实例，提供奖励计算和持仓映射。
            gamma: 折扣因子 γ，默认 0.99（与论文 Algorithm 1 一致）。
            result_dir: 轨迹缓存输出目录根路径。
            sampling_seed: Phase I 轨迹采样随机种子，用于结果复现。
        """
        self.env = env
        self.pair = env.pair
        self.horizon = env.horizon
        self.m = env.m  # 最大持仓量
        self.gamma = gamma
        self.result_dir = result_dir
        self.sampling_seed = sampling_seed

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    @staticmethod
    def build_trajectory_cache_path(result_dir: str, pair: str) -> str:
        """构造 Phase I 轨迹缓存路径。

        新增该方法的目的是统一训练脚本和 DP 规划器对缓存路径的定义，
        避免“保存路径”和“加载路径”不一致导致重复生成轨迹。

        Args:
            result_dir: 结果输出目录根路径
            pair: 交易对名称

        Returns:
            统一的轨迹缓存文件路径
        """
        return os.path.join(result_dir, pair, "dp_trajectories", "trajectories.npz")

    def plan(
        self,
        states: pl.DataFrame,
        prices: np.ndarray,
        return_debug_info: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """对单个 horizon 执行动态规划，生成最优示范轨迹。

        # Algorithm 1, Step 1: 初始化
        # Algorithm 1, Step 2: 反向填表 (Backward pass)
        # Algorithm 1, Step 3: 前向追踪 (Forward pass)

        Args:
            states: 单个 horizon 的状态序列, polars DataFrame
            prices: 单个 horizon 的价格序列, shape (N,) 或 (N+1,)
                    若只提供 N 个价格，最后一步价差按 0 处理。
            return_debug_info: 是否返回 DP 调试信息。
                               开启后会额外返回 V 表和 Pi 表，便于验证 Bellman 一致性。

        Returns:
            默认返回:
                s_demo: 状态序列 (N, state_dim)
                a_demo: 动作序列 (N,) 值域 {0, 1, 2}
                r_demo: 奖励序列 (N,)
            当 return_debug_info=True 时，额外返回 debug_info 字典。
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
                        # c=0 表示尚未发生动作变化
                        # c=1 表示已经发生过一次动作变化
                        next_position = TradingEnv.POSITION_MAP[a_next] * self.m

                        # 计算约束转移
                        c_next = self._compute_next_constraint(c, a_prev, a_next)
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

        # 初始状态: flat, 未发生动作变化
        current_action_idx = self.FLAT_ACTION  # flat
        current_c = 0  # 未发生动作变化
        current_position = 0  # flat 持仓

        # Algorithm 1, Step 9-12: 前向追踪 t = 0 到 N-2
        for t in range(N - 1):
            # 选择最优动作
            next_action = int(Pi[t, current_action_idx, current_c])
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
            next_c = self._compute_next_constraint(
                current_c, current_action_idx, next_action
            )
            if next_c < 0:
                raise RuntimeError(
                    "前向追踪阶段命中了非法动作迁移，这说明 DP 表与约束转移不一致。"
                )
            current_c = next_c
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
        if not return_debug_info:
            return s_demo, a_demo, r_demo

        debug_info: Dict[str, Any] = {
            "V": V,
            "Pi": Pi,
            "gamma": np.float64(self.gamma),
            "horizon": np.int64(N),
        }
        return s_demo, a_demo, r_demo, debug_info

    def generate_trajectories(
        self, num_trajectories: int = 30000
    ) -> Dict[str, np.ndarray]:
        """批量生成示范轨迹。

        按论文描述从训练集采样固定长度为 h 的数据块，再对每个采样块执行 DP 规划。
        当可用起点不足目标数量时，才退化为有放回采样；否则默认无放回采样。

        Args:
            num_trajectories: 目标轨迹数量，默认 30000。

        Returns:
            字典包含:
                'states':  (num_trajectories, h, state_dim)
                'actions': (num_trajectories, h)
                'rewards': (num_trajectories, h)
                'sampled_start_indices': (num_trajectories,)
                'sampling_seed': 标量
                'replace': 标量
                'num_available_starts': 标量
                'num_sampled_trajectories': 标量
                'pair': 交易对名称
                'horizon': 轨迹长度
                'gamma': 折扣因子
        """
        if self.env.states_dataframe is None:
            raise ValueError(
                f"DPPlanner.generate_trajectories() 需要 env.states_dataframe，"
                f"但当前为 None。请在创建 TradingEnv 时传入 states_dataframe 参数。"
            )

        valid_start_indices = self._get_valid_start_indices()
        num_available_starts = len(valid_start_indices)
        if num_available_starts == 0:
            logger.warning("环境中无可用 horizon，生成全 flat 轨迹")
            result = self._generate_all_flat(num_trajectories)
            result.update(
                {
                    "sampled_start_indices": np.full(num_trajectories, -1, dtype=np.int64),
                    "sampling_seed": np.int64(self.sampling_seed),
                    "replace": np.bool_(False),
                    "num_available_starts": np.int64(0),
                    "num_sampled_trajectories": np.int64(num_trajectories),
                    "pair": np.array(self.pair),
                    "horizon": np.int64(self.horizon),
                    "gamma": np.float64(self.gamma),
                    "training_rows": np.int64(len(self.env.states)),
                    "state_dim": np.int64(self.env.state_dim),
                    "commission_rate": np.float64(self.env.commission_rate),
                    "max_position": np.int64(self.m),
                    "algorithm_variant": np.array("paper_single_change"),
                }
            )
            self._save_trajectories(result)
            return result

        replace = num_available_starts < num_trajectories
        sampled_start_indices = self._sample_start_indices(
            valid_start_indices=valid_start_indices,
            num_trajectories=num_trajectories,
            replace=replace,
        )

        logger.info(
            "开始生成 DP 示范轨迹: pair=%s, num_available_starts=%d, target=%d, replace=%s, seed=%d",
            self.pair,
            num_available_starts,
            num_trajectories,
            replace,
            self.sampling_seed,
        )

        all_states = []
        all_actions = []
        all_rewards = []

        # 使用 tqdm 显示进度条
        for i, start in enumerate(tqdm(sampled_start_indices, desc="生成DP轨迹")):
            start = int(start)
            end = start + self.horizon
            h_states = self.env.states_dataframe.slice(start, self.horizon)
            # 价格需要多取一个点用于最后一步奖励计算
            price_end = min(end + 1, len(self.env.prices))
            h_prices = self.env.prices[start:price_end]

            s_demo, a_demo, r_demo = self.plan(h_states, h_prices)

            if i == 0:
                logger.info(
                    "Sample 0 shapes: h_states=%s, h_prices=%s, s_demo=%s, a_demo=%s, r_demo=%s",
                    h_states.shape,
                    h_prices.shape,
                    s_demo.shape,
                    a_demo.shape,
                    r_demo.shape,
                )

            all_states.append(s_demo)
            all_actions.append(a_demo)
            all_rewards.append(r_demo)

        result = {
            # states 保持 float32 以控制轨迹缓存体积；rewards 保持 float64 以便严格验证 Eq.(1) 回放一致性。
            "states": np.array(all_states, dtype=np.float32),
            "actions": np.array(all_actions, dtype=np.int32),
            "rewards": np.array(all_rewards, dtype=np.float64),
            "sampled_start_indices": sampled_start_indices.astype(np.int64),
            "sampling_seed": np.int64(self.sampling_seed),
            "replace": np.bool_(replace),
            "num_available_starts": np.int64(num_available_starts),
            "num_sampled_trajectories": np.int64(num_trajectories),
            "pair": np.array(self.pair),
            "horizon": np.int64(self.horizon),
            "gamma": np.float64(self.gamma),
            "training_rows": np.int64(len(self.env.states)),
            "state_dim": np.int64(self.env.state_dim),
            "commission_rate": np.float64(self.env.commission_rate),
            "max_position": np.int64(self.m),
            "algorithm_variant": np.array("paper_single_change"),
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

    def _get_valid_start_indices(self) -> np.ndarray:
        """返回所有可用于固定长度 horizon 采样的起点索引。

        论文描述是“sample n data chunks of fixed length h”。
        因此这里不再按非重叠 horizon 切块，而是使用滑窗方式枚举
        所有长度恰好为 h 的合法起点，以保证 demonstration 的多样性。

        Returns:
            所有合法起点组成的一维整数数组
        """
        total_steps = len(self.env.states)
        max_start = total_steps - self.horizon
        if max_start < 0:
            return np.empty(0, dtype=np.int64)
        return np.arange(max_start + 1, dtype=np.int64)

    def _sample_start_indices(
        self,
        valid_start_indices: np.ndarray,
        num_trajectories: int,
        replace: bool,
    ) -> np.ndarray:
        """从合法起点集合中采样固定数量的 trajectory 起点。

        Args:
            valid_start_indices: 全部合法起点
            num_trajectories: 目标采样条数
            replace: 是否有放回采样

        Returns:
            采样后的起点索引数组
        """
        rng = np.random.default_rng(self.sampling_seed)
        sampled = rng.choice(
            valid_start_indices,
            size=num_trajectories,
            replace=replace,
        )
        return np.asarray(sampled, dtype=np.int64)

    def _compute_next_constraint(
        self, c: int, a_prev: int, a_next: int
    ) -> int:
        """计算约束标志的转移。

        严格按照论文 Algorithm 1 中的约束项实现：
        c_next = c + 1[a_prev != a_next]，且要求 c_next <= 1。

        因此该实现允许的轨迹类型为：
          flat → flat → ...（从不交易）
          flat → long → long → ...（只发生一次动作变化）
          flat → short → short → ...（只发生一次动作变化）
          long → long → ... / short → short → ...（对一般状态也兼容）

        不允许的轨迹类型为：
          flat → long → flat（两次动作变化）
          flat → short → flat（两次动作变化）
          short → long / long → short（动作再次变化）

        Args:
            c: 当前约束标志 (0 或 1)
            a_prev: 上一步动作（代表当前持仓方向）
            a_next: 当前动作（代表目标持仓方向）

        Returns:
            新的约束标志 (0 或 1)，-1 表示违反约束。
        """
        c_next = c + int(a_prev != a_next)
        if c_next > 1:
            return -1
        return c_next

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
        rewards = np.zeros((num_trajectories, h), dtype=np.float64)

        return {"states": states, "actions": actions, "rewards": rewards}

    def _save_trajectories(self, trajectories: Dict[str, np.ndarray]) -> None:
        """保存轨迹到统一的 Phase I 缓存路径。

        Args:
            trajectories: 轨迹字典，除 states/actions/rewards 外，也可包含采样元数据。
        """
        save_path = self.build_trajectory_cache_path(self.result_dir, self.pair)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **trajectories)
        logger.info("轨迹已保存到 %s", save_path)
