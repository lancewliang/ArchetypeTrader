"""Single-trade DP planner（论文 Algorithm 1）.

设计文档锚点: §5。

DP 状态: ``(t, action, changed)``，``action ∈ {0, 1, 2}``，``changed ∈ {0, 1}``。
转移约束: ``c + 1[i ≠ j] ≤ 1``，即 horizon 内最多一次切换。

末步处理（关键修复点）:
- DP 主循环只填 ``t ∈ [0, N-2]``。
- ``actions[N-1] = actions[N-2]``，不计入 ``num_switches``。
- 末步 reward 仍按 reward_alignment 用 markout 行结算。

实现要点:
1. 预构造 ``transition_reward[h, 3, 3]`` 与 ``transition_valid[h, 3, 3]``，避免 DP 内部反复
   调 cost_model（昂贵）。
2. 反向递推填 ``V[t, action, changed]``、``Π[t, action, changed]``。
3. 正向回溯生成 ``actions``；末步按论文复制。
4. 用 env replay 复算 rewards，保证 DP / replay 完全一致。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from src.trading.cost_model import ExecutionBook, ExecutionResult, LobDepthCostModel
from src.trading.env import HorizonInputs, TradingEnv
from src.trading.reward_alignment import RewardAlignment


_NEG_INF = -1e18


@dataclass(frozen=True)
class DPInputs:
    """单 horizon 的 DP 输入。"""
    prices: List[float]
    execution_books: List[ExecutionBook]
    horizon: int


@dataclass
class DPResult:
    """单 horizon 的 DP 输出。"""
    actions: List[int]
    rewards: List[float]
    total_return: float
    num_switches: int
    is_no_trade: bool
    reject_events: List[dict]


class SingleTradeDPPlanner:
    """对单个 horizon 求解 single-trade DP。"""

    NUM_ACTIONS = 3  # short / flat / long
    INITIAL_ACTION = 1  # flat 起始（论文设定）

    def __init__(
        self,
        cost_model: LobDepthCostModel,
        reward_alignment: RewardAlignment,
        max_position: int = 1,
        gamma: float = 1.0,
    ) -> None:
        self.cost_model = cost_model
        self.reward_alignment = reward_alignment
        self.max_position = max_position
        self.gamma = gamma

    def plan(self, inputs: DPInputs) -> DPResult:
        """求解单个 horizon。

        Steps
        -----
        1. ``_precompute_transitions``: 对每个 ``t`` 与每对 ``(prev_a, target_a)``
           预算 ``transition_reward[h, 3, 3]`` 与 ``transition_valid[h, 3, 3]``，
           DP 反向递推时只查表，避免反复调用 cost_model。
        2. ``_backward``: 反向递推填 ``V[t, action, changed]``、``Π[t, action, changed]``；
           ``t ∈ [0, h-1]`` 全部填充（``V[h]=0`` 作为终止），保证 V[h-2] 能感知末步收益。
        3. ``_forward``: 正向回溯 actions（从 ``flat, changed=0`` 起，仅遍历 ``t=0..h-2``）；
           末步按论文 Algorithm 1 第 13 行复制 ``actions[N-1] = actions[N-2]``。
        4. ``_replay``: 把 actions 喂回 env 复算 rewards 与 reject events，
           保证 DP teacher 与 student replay 使用同一套 reward / cost 语义。
        5. 统计 ``num_switches`` (不含末步复制) 与 ``is_no_trade`` (全 flat)。

        Returns
        -------
        DPResult : 包含 ``actions / rewards / total_return / num_switches /
                   is_no_trade / reject_events``。

        Raises
        ------
        ValueError : ``inputs.horizon <= 0``。
        """
        if inputs.horizon <= 0:
            raise ValueError(f"horizon 必须 > 0, got {inputs.horizon}")
        # 预算单步转移 reward 与 valid。
        rewards_table, valid_table, exec_results = self._precompute_transitions(inputs)
        V, Pi = self._backward(rewards_table, valid_table, inputs.horizon)
        actions = self._forward(Pi, inputs.horizon)

        # 末步: actions[N-1] = actions[N-2]（论文 Algorithm 1 第 13 行）。
        if inputs.horizon >= 2:
            actions[-1] = actions[-2]

        # 用 env 复算 reward，确保 DP / replay 完全一致。
        rewards, reject_events = self._replay(actions, inputs)
        total_return = sum(rewards)

        # num_switches 不含末步复制 → 在序列 [0..N-2] 上数切换次数。
        num_switches = 0
        for i in range(1, inputs.horizon - 1):
            if actions[i] != actions[i - 1]:
                num_switches += 1
        is_no_trade = all(a == self.INITIAL_ACTION for a in actions)
        return DPResult(
            actions=actions,
            rewards=rewards,
            total_return=total_return,
            num_switches=num_switches,
            is_no_trade=is_no_trade,
            reject_events=reject_events,
        )

    # ---------- 私有 ----------

    def _precompute_transitions(self, inputs: DPInputs):
        """对每个 ``t`` 与每对 ``(prev_action, target_action)`` 预算 reward 与 valid。

        - 不换仓（``prev == target``）: reward = ``target_position * (markout - exec)``，
          直接给 valid=True，不调 cost_model。
        - 换仓: 调 ``cost_model.execute``；rejected → valid=False，reward=-inf
          （DP backward 时该 transition 不会被选中）；否则 reward 扣除 fee + slippage。

        Returns
        -------
        rewards_table : List[List[List[float]]] shape [h, 3, 3]
        valid_table   : List[List[List[bool]]]  shape [h, 3, 3]
        exec_results  : 同形状的 ExecutionResult 表，仅供调试 / 单测，
                         DP 表本身只看 reward + valid。
        """
        h = inputs.horizon
        rewards_table = [
            [[0.0 for _ in range(self.NUM_ACTIONS)] for _ in range(self.NUM_ACTIONS)]
            for _ in range(h)
        ]
        valid_table = [
            [[True for _ in range(self.NUM_ACTIONS)] for _ in range(self.NUM_ACTIONS)]
            for _ in range(h)
        ]
        exec_results: List[List[List[Optional[ExecutionResult]]]] = [
            [[None for _ in range(self.NUM_ACTIONS)] for _ in range(self.NUM_ACTIONS)]
            for _ in range(h)
        ]

        for t in range(h):
            rows = self.reward_alignment.rows(t)
            book = inputs.execution_books[t]
            p_exec = float(inputs.prices[rows.execution_row])
            p_markout = float(inputs.prices[rows.markout_row])
            for prev_a in range(self.NUM_ACTIONS):
                prev_pos = self.max_position * (prev_a - 1)
                for target_a in range(self.NUM_ACTIONS):
                    target_pos = self.max_position * (target_a - 1)
                    if prev_pos == target_pos:
                        # 不换仓: 收益 = position * (markout - exec)，无成本。
                        reward = target_pos * (p_markout - p_exec)
                        rewards_table[t][prev_a][target_a] = reward
                        valid_table[t][prev_a][target_a] = True
                        continue
                    res = self.cost_model.execute(
                        prev_position=prev_pos,
                        target_position=target_pos,
                        execution_book=book,
                    )
                    exec_results[t][prev_a][target_a] = res
                    if res.rejected:
                        valid_table[t][prev_a][target_a] = False
                        rewards_table[t][prev_a][target_a] = _NEG_INF
                        continue
                    pos_after = res.filled_position
                    reward = pos_after * (p_markout - p_exec) - res.cost
                    rewards_table[t][prev_a][target_a] = reward
                    valid_table[t][prev_a][target_a] = True
        return rewards_table, valid_table, exec_results

    def _backward(self, rewards_table, valid_table, horizon: int):
        """反向 DP（严格对齐论文 Algorithm 1 line 2）。

        - ``V[t, action, changed]`` = 从状态 ``(action, changed)`` 在时间 ``t``
          出发的最优期望价值；``V[h, *, *] = 0``（终止）。
        - 转移约束: ``c + 1[i ≠ j] ≤ 1``，即 horizon 内最多一次切换。
        - DP 主循环填 ``t ∈ [0, h-1]``；``V[h-1]`` 使用 ``V[h]=0`` 作为终止递推得到，
          ``V[t < h-1]`` 使用 ``V[t+1]``。Pi[h-1] 计算后不被 forward 使用
          （forward 仅遍历 t=0..h-2，末步由 plan() 复制 actions[h-2] 得到），
          但 V[h-1] 必须正确填充，否则 V[h-2] 会丢失末步收益贡献。
        - 当所有候选 target 都被 ``valid=False`` 拒绝时，``V[t, prev, c]`` 视为 0
          （等价于强制保留 prev 仓位；与论文 Algorithm 1 实质等价）。
        """
        # 形状 [h+1, 3, 2]
        V = [
            [[0.0 for _ in range(2)] for _ in range(self.NUM_ACTIONS)]
            for _ in range(horizon + 1)
        ]
        # Pi[t, action, changed] = 下一步动作（target_action）；t ∈ [0, h-1] 都填充。
        Pi = [
            [[0 for _ in range(2)] for _ in range(self.NUM_ACTIONS)]
            for _ in range(horizon)
        ]
        # 论文 Algorithm 1 line 2: for t = N-1 downto 0.
        # 即 t = h-1 也参与最优化以确保 V[h-1] 反映末步收益。
        for t in range(horizon - 1, -1, -1):
            for prev_a in range(self.NUM_ACTIONS):
                for c in range(2):
                    best_val = _NEG_INF
                    best_action = prev_a  # 默认保持
                    for target_a in range(self.NUM_ACTIONS):
                        switch = 1 if target_a != prev_a else 0
                        new_c = c + switch
                        if new_c > 1:
                            # single-trade 约束被违反
                            continue
                        if not valid_table[t][prev_a][target_a]:
                            continue
                        immediate = rewards_table[t][prev_a][target_a]
                        future = V[t + 1][target_a][new_c]
                        val = immediate + self.gamma * future
                        if val > best_val:
                            best_val = val
                            best_action = target_a
                    V[t][prev_a][c] = best_val if best_val > _NEG_INF / 2 else 0.0
                    Pi[t][prev_a][c] = best_action
        return V, Pi

    def _forward(self, Pi, horizon: int) -> List[int]:
        """从 ``(prev_action=flat, changed=0)`` 起回溯 actions。

        - 反复读 ``Pi[t][prev_a][c]`` 拿到最优 target action，更新 ``(prev_a, c)``。
        - ``actions[horizon - 1]`` 在 ``plan()`` 末尾复制为 ``actions[horizon - 2]``，
          这里只填 ``[0, horizon - 2)``。
        """
        actions: List[int] = [self.INITIAL_ACTION] * horizon
        prev_a = self.INITIAL_ACTION
        c = 0
        for t in range(horizon - 1):
            target_a = Pi[t][prev_a][c]
            actions[t] = target_a
            if target_a != prev_a:
                c = min(1, c + 1)
            prev_a = target_a
        # actions[horizon - 1] 在 plan() 中复制 actions[horizon - 2]
        return actions

    def _replay(self, actions: List[int], inputs: DPInputs):
        """把 actions 喂进 env 复算 rewards 与 reject events，保证 DP / replay 完全一致。

        重要: 不直接用 ``rewards_table`` 累加，而是再走一次 env，确保以下不变量:
        - DP teacher 的 ``rewards`` 与 student replay 在同一份 cost_config 下完全可比。
        - reject_events 通过 env.StepInfo 收集，便于 demo_generator 汇总
          ``dataset_reject_rate`` / ``per_horizon_reject_rate``。
        """
        env = TradingEnv(
            cost_model=self.cost_model,
            reward_alignment=self.reward_alignment,
            max_position=self.max_position,
        )
        env.reset(
            HorizonInputs(prices=inputs.prices, execution_books=inputs.execution_books)
        )
        rewards, infos = env.replay(actions)
        reject_events = [info.reject_event for info in infos if info.reject_event is not None]
        return rewards, reject_events
