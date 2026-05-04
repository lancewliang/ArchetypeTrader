"""Single-trade DP planner（论文 Algorithm 1）.

设计文档锚点: §5。

DP 状态: ``(t, action, changed)``，``action ∈ {0, 1, 2}``，``changed ∈ {0, 1}``。
转移约束: ``c + 1[i ≠ j] ≤ 1``，即 horizon 内最多一次切换。

末步处理（关键修复点）:
- DP 在 ``t=N-1`` 只允许保持上一仓位，不允许把切换推迟到末步。
- ``actions[N-1] = actions[N-2]``，不计入 ``num_switches``。
- 末步 reward 仍按 reward_alignment 用 markout 行结算。

实现要点:
1. 预构造 ``transition_reward[h, 3, 3]`` 与 ``transition_valid[h, 3, 3]``，避免 DP 内部反复
   调 cost_model（昂贵）。
2. 反向递推填 ``V[t, action, changed]``、``Π[t, action, changed]``。
3. 正向回溯生成 ``actions``；末步按论文复制。
4. 用预计算转移表回放 rewards；异常 invalid 动作才回退 env replay。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

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
    precompute_evaluated_count: int = 0
    precompute_rejected_count: int = 0
    precompute_rejected_by_pair: dict | None = None


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
           ``t ∈ [0, h-1]`` 全部填充（``V[h]=0`` 作为终止），但末步只允许保持
           当前仓位，保证 V[h-2] 能感知末步持仓收益且不会依赖不可执行切换。
        3. ``_forward``: 正向回溯 actions（从 ``flat, changed=0`` 起，仅遍历 ``t=0..h-2``）；
           末步按论文 Algorithm 1 第 13 行复制 ``actions[N-1] = actions[N-2]``。
        4. ``_replay``: 用同一转移表快速回放 rewards；若发现 invalid 动作，
           回退 env replay，保证 DP teacher 与 student replay 使用同一套 reward / cost 语义。
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
        (
            rewards_table,
            valid_table,
            _exec_results,
            precompute_evaluated_count,
            precompute_rejected_count,
            precompute_rejected_by_pair,
        ) = self._precompute_transitions(inputs)
        V, Pi = self._backward(rewards_table, valid_table, inputs.horizon)
        actions = self._forward(Pi, inputs.horizon)

        # 末步: actions[N-1] = actions[N-2]（论文 Algorithm 1 第 13 行）。
        if inputs.horizon >= 2:
            actions[-1] = actions[-2]

        # 用同一份预计算表回放 reward；若异常选中 invalid transition，则回退 env。
        rewards, reject_events = self._replay(
            actions, inputs, rewards_table=rewards_table, valid_table=valid_table
        )
        total_return = float(np.sum(rewards, dtype=np.float64))

        # num_switches 不含末步复制 → 在序列 [0..N-2] 上数切换次数。
        if inputs.horizon > 2:
            action_arr = np.asarray(actions[:-1], dtype=np.int8)
            num_switches = int(np.count_nonzero(action_arr[1:] != action_arr[:-1]))
        else:
            num_switches = 0
        is_no_trade = all(a == self.INITIAL_ACTION for a in actions)
        return DPResult(
            actions=actions,
            rewards=rewards,
            total_return=total_return,
            num_switches=num_switches,
            is_no_trade=is_no_trade,
            reject_events=reject_events,
            precompute_evaluated_count=precompute_evaluated_count,
            precompute_rejected_count=precompute_rejected_count,
            precompute_rejected_by_pair=precompute_rejected_by_pair,
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
        rewards_table : np.ndarray shape [h, 3, 3]
        valid_table   : np.ndarray shape [h, 3, 3]
        exec_results  : 标量 fallback 返回同形状的 ExecutionResult 表；
                         numpy 快路径返回 None，DP 表本身只看 reward + valid。
        """
        if type(self.cost_model) is not LobDepthCostModel:
            return self._precompute_transitions_scalar(inputs)

        h = inputs.horizon
        prices = np.asarray(inputs.prices, dtype=np.float64)
        if self.reward_alignment.mode == "paper_formula":
            execution_rows = np.arange(h, dtype=np.int64)
            markout_rows = execution_rows + 1
        else:
            execution_rows = np.arange(h, dtype=np.int64) + 1
            markout_rows = execution_rows + 1
        p_exec = prices[execution_rows]
        p_markout = prices[markout_rows]
        price_delta = p_markout - p_exec

        positions = (
            np.arange(self.NUM_ACTIONS, dtype=np.int64) - self.INITIAL_ACTION
        ) * int(self.max_position)
        prev_positions = positions[:, None]
        target_positions = positions[None, :]
        deltas = target_positions - prev_positions
        switch_mask = deltas != 0

        rewards_table = np.broadcast_to(
            price_delta[:, None, None] * positions[None, None, :],
            (h, self.NUM_ACTIONS, self.NUM_ACTIONS),
        ).astype(np.float64, copy=True)
        valid_table = np.ones(
            (h, self.NUM_ACTIONS, self.NUM_ACTIONS), dtype=np.bool_
        )

        if h == 0:
            return rewards_table, valid_table, None, 0, 0, {}

        ask_prices = np.asarray(
            [
                list(book.ask_prices)[: self.cost_model.book_levels]
                for book in inputs.execution_books
            ],
            dtype=np.float64,
        )
        ask_sizes = np.asarray(
            [
                list(book.ask_sizes)[: self.cost_model.book_levels]
                for book in inputs.execution_books
            ],
            dtype=np.float64,
        )
        bid_prices = np.asarray(
            [
                list(book.bid_prices)[: self.cost_model.book_levels]
                for book in inputs.execution_books
            ],
            dtype=np.float64,
        )
        bid_sizes = np.asarray(
            [
                list(book.bid_sizes)[: self.cost_model.book_levels]
                for book in inputs.execution_books
            ],
            dtype=np.float64,
        )
        mark_prices = np.asarray(
            [float(book.mark_price) for book in inputs.execution_books],
            dtype=np.float64,
        )

        rejected_by_pair: dict[str, int] = {}
        rejected_count = 0
        for prev_a in range(self.NUM_ACTIONS):
            prev_pos = int(positions[prev_a])
            for target_a in range(self.NUM_ACTIONS):
                target_pos = int(positions[target_a])
                delta = target_pos - prev_pos
                if delta == 0:
                    continue
                need = float(abs(delta))
                if delta > 0:
                    level_prices = ask_prices
                    level_sizes = ask_sizes
                else:
                    level_prices = bid_prices
                    level_sizes = bid_sizes

                positive_sizes = np.maximum(level_sizes, 0.0)
                cumulative_before = np.cumsum(positive_sizes, axis=1) - positive_sizes
                take = np.clip(need - cumulative_before, 0.0, positive_sizes)
                filled_qty = np.sum(take, axis=1)
                weighted_price_sum = np.sum(take * level_prices, axis=1)
                valid = filled_qty + 1e-9 >= need
                valid_table[:, prev_a, target_a] = valid

                fill_price = np.divide(
                    weighted_price_sum,
                    filled_qty,
                    out=np.zeros_like(weighted_price_sum),
                    where=filled_qty > 0,
                )
                fee = self.cost_model.commission_rate * need * mark_prices
                slippage = (
                    need
                    * np.abs(fill_price - mark_prices)
                    * self.cost_model.slippage_multiplier
                )
                cost = fee + slippage
                rewards = target_pos * price_delta - cost
                rewards_table[:, prev_a, target_a] = np.where(valid, rewards, _NEG_INF)

                reject_n = int(np.count_nonzero(~valid))
                if reject_n:
                    rejected_by_pair[f"{prev_pos:+d}->{target_pos:+d}"] = reject_n
                    rejected_count += reject_n

        evaluated_count = int(h * np.count_nonzero(switch_mask))
        return (
            rewards_table,
            valid_table,
            None,
            evaluated_count,
            rejected_count,
            rejected_by_pair,
        )

    def _precompute_transitions_scalar(self, inputs: DPInputs):
        """标量 fallback，保留自定义 cost_model.execute 的语义。"""
        h = inputs.horizon
        rewards_table = np.zeros(
            (h, self.NUM_ACTIONS, self.NUM_ACTIONS), dtype=np.float64
        )
        valid_table = np.ones(
            (h, self.NUM_ACTIONS, self.NUM_ACTIONS), dtype=np.bool_
        )
        exec_results: List[List[List[Optional[ExecutionResult]]]] = [
            [[None for _ in range(self.NUM_ACTIONS)] for _ in range(self.NUM_ACTIONS)]
            for _ in range(h)
        ]
        rejected_by_pair: dict[str, int] = {}
        evaluated_count = 0
        rejected_count = 0

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
                        rewards_table[t, prev_a, target_a] = target_pos * (
                            p_markout - p_exec
                        )
                        valid_table[t, prev_a, target_a] = True
                        continue
                    evaluated_count += 1
                    res = self.cost_model.execute(
                        prev_position=prev_pos,
                        target_position=target_pos,
                        execution_book=book,
                    )
                    exec_results[t][prev_a][target_a] = res
                    if res.rejected:
                        valid_table[t, prev_a, target_a] = False
                        rewards_table[t, prev_a, target_a] = _NEG_INF
                        rejected_count += 1
                        key = f"{prev_pos:+d}->{target_pos:+d}"
                        rejected_by_pair[key] = rejected_by_pair.get(key, 0) + 1
                        continue
                    reward = res.filled_position * (p_markout - p_exec) - res.cost
                    rewards_table[t, prev_a, target_a] = reward
                    valid_table[t, prev_a, target_a] = True
        return (
            rewards_table,
            valid_table,
            exec_results,
            evaluated_count,
            rejected_count,
            rejected_by_pair,
        )

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
        rewards_np = np.asarray(rewards_table, dtype=np.float64)
        valid_np = np.asarray(valid_table, dtype=np.bool_)
        V = np.zeros((horizon + 1, self.NUM_ACTIONS, 2), dtype=np.float64)
        Pi = np.zeros((horizon, self.NUM_ACTIONS, 2), dtype=np.int8)
        action_ids = np.arange(self.NUM_ACTIONS, dtype=np.int8)
        switch_mask = action_ids[None, :] != action_ids[:, None]
        # 论文 Algorithm 1 line 2: for t = N-1 downto 0.
        # 即 t = h-1 也参与最优化以确保 V[h-1] 反映末步收益。
        for t in range(horizon - 1, -1, -1):
            rewards_t = rewards_np[t]
            valid_t = valid_np[t]
            next_v = V[t + 1]
            diagonal_rewards = rewards_t[action_ids, action_ids]

            hold_vals = diagonal_rewards + self.gamma * next_v[:, 1]
            V[t, :, 1] = np.where(valid_t[action_ids, action_ids], hold_vals, 0.0)
            Pi[t, :, 1] = action_ids

            if t == horizon - 1:
                terminal_vals = diagonal_rewards + self.gamma * next_v[:, 0]
                V[t, :, 0] = np.where(
                    valid_t[action_ids, action_ids], terminal_vals, 0.0
                )
                Pi[t, :, 0] = action_ids
                continue

            future_c0 = np.where(
                switch_mask,
                next_v[:, 1][None, :],
                next_v[:, 0][None, :],
            )
            candidates = rewards_t + self.gamma * future_c0
            candidates = np.where(valid_t, candidates, _NEG_INF)
            best_actions = np.argmax(candidates, axis=1).astype(np.int8)
            best_vals = candidates[action_ids, best_actions]
            V[t, :, 0] = np.where(best_vals > _NEG_INF / 2, best_vals, 0.0)
            Pi[t, :, 0] = best_actions
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
            target_a = int(Pi[t, prev_a, c])
            actions[t] = target_a
            if target_a != prev_a:
                c = min(1, c + 1)
            prev_a = target_a
        # actions[horizon - 1] 在 plan() 中复制 actions[horizon - 2]
        return actions

    def _replay(
        self,
        actions: List[int],
        inputs: DPInputs,
        *,
        rewards_table=None,
        valid_table=None,
    ):
        """把 actions 回放成 rewards 与 reject events，保证 DP / replay 完全一致。

        重要: 快路径读取同一份 ``rewards_table``，它由同一 cost_model 公式预计算；
        若发现所选动作含 invalid transition，则回退到 env 逐步 replay，确保以下不变量:
        - DP teacher 的 ``rewards`` 与 student replay 在同一份 cost_config 下完全可比。
        - reject_events 通过 env.StepInfo 收集，便于 demo_generator 汇总
          ``dataset_reject_rate`` / ``per_horizon_reject_rate``。
        """
        if rewards_table is not None and valid_table is not None:
            prev_actions = np.empty(len(actions), dtype=np.int64)
            prev_actions[0] = self.INITIAL_ACTION
            if len(actions) > 1:
                prev_actions[1:] = np.asarray(actions[:-1], dtype=np.int64)
            target_actions = np.asarray(actions, dtype=np.int64)
            t_idx = np.arange(len(actions), dtype=np.int64)
            valid = np.asarray(valid_table, dtype=np.bool_)[
                t_idx, prev_actions, target_actions
            ]
            if bool(np.all(valid)):
                rewards = np.asarray(rewards_table, dtype=np.float64)[
                    t_idx, prev_actions, target_actions
                ]
                return rewards.astype(float).tolist(), []

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
