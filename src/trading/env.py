"""统一的交易环境（合并自原 ``src/envs/trading_env.py``）.

设计文档锚点: §4.9。

reward 计算（论文公式重申）::

    r_t = P_t * (p_markout - p_exec) - O_t

其中 ``P_t`` 是动作映射后的目标仓位；``O_t = fee + slippage`` 由 cost_model 给出。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .cost_model import ExecutionBook, ExecutionResult, LobDepthCostModel
from .reward_alignment import AlignmentRows, RewardAlignment


@dataclass(frozen=True)
class HorizonInputs:
    """单 horizon 的 env 输入。

    长度约定:
    - ``prices``: ``h + lookahead`` 个 mark 价格；paper_formula 时 lookahead=1，
      next_row_execution 时 lookahead=2。
    - ``execution_books``: 必须能覆盖所有 ``execution_row``。在 ``next_row_execution``
      模式下，``execution_books[t]`` 实际对应数据中的 ``t+1`` 行盘口（构建时已对齐）。
    """
    prices: Sequence[float]
    execution_books: Sequence[ExecutionBook]


@dataclass
class StepInfo:
    """``TradingEnv.step()`` 返回的诊断信息。

    Attributes
    ----------
    fee, slippage : 来自 ``LobDepthCostModel``，便于汇总 ``cost_paid`` 指标。
    fill_price : 加权成交价；``None`` 表示未换仓或被 reject。
    filled_position : 实际成交后持仓；reject 时等于 ``prev_position``。
    nav : 累计净值（self._nav，自 reset 起的 reward 总和）。
    execution_row, markout_row : 由 ``RewardAlignment`` 给出的实际行号，便于审计。
    rejected : 当前 step 是否触发 reject_transition。
    reject_event : 用于 demo_generator 与 replay evaluator 收集统计的字典；
                   ``None`` 表示无事件。
    """
    fee: float
    slippage: float
    fill_price: Optional[float]
    filled_position: int
    nav: float
    execution_row: int
    markout_row: int
    rejected: bool
    reject_event: Optional[dict]


class TradingEnv:
    """分钟级单资产 env。

    生命周期:
    1. ``reset(horizon, initial_position=0)``: 注入 horizon 输入并重置内部
       ``step``、``position``、``cash``、``nav``。
    2. 反复 ``step(action)`` 直到 ``done=True``。
    3. ``replay(actions)``: 一次执行整段动作序列，返回 step rewards 与
       ``StepInfo`` 列表（等价于反复调用 ``step``）。

    边界:
    - action 空间 ``{0, 1, 2}``；映射 ``position = max_position * (action - 1)``。
    - 不依赖任何全局 state；多个实例可并发（DP 并行）。
    - 内部经由 ``LobDepthCostModel`` 与 ``RewardAlignment``，绝不自实现成本/行号。
    """

    def __init__(
        self,
        cost_model: LobDepthCostModel,
        reward_alignment: RewardAlignment,
        max_position: int = 1,
    ) -> None:
        self.cost_model = cost_model
        self.reward_alignment = reward_alignment
        self.max_position = max_position
        self._horizon: Optional[HorizonInputs] = None
        self._t: int = 0
        self._position: int = 0
        self._nav: float = 0.0

    def reset(self, horizon: HorizonInputs, initial_position: int = 0) -> None:
        """注入 horizon、设置初始仓位。

        ``initial_position`` 默认 0；Phase II 跨 horizon 衔接时传入非零值，
        env 必须在第一步把 ``initial_position -> first_target_position`` 的
        换仓成本通过同一 ``CostModel`` 扣掉，避免逐 horizon 独立选 archetype
        系统性低估边界成本。

        Raises
        ------
        ValueError : ``|initial_position| > max_position``。
        """
        if not -self.max_position <= initial_position <= self.max_position:
            raise ValueError(
                f"initial_position {initial_position} 越界，必须 ∈ [-{self.max_position}, {self.max_position}]"
            )
        self._horizon = horizon
        self._t = 0
        self._position = initial_position
        self._nav = 0.0

    def step(self, action: int) -> tuple[float, bool, StepInfo]:
        """执行单步动作。

        计算路径
        --------
        1. ``action -> target_position`` 映射。
        2. ``cost_model.execute`` 估算 fee/slippage/filled_position（可能被 reject）。
        3. ``reward = filled_position * (p_markout - p_exec) - cost``。
           - rejected 时 ``filled_position == prev_position``，reward 仅含持仓收益。
        4. 推进 ``self._t``；当 ``t >= len(execution_books)`` 时 ``done=True``。

        Returns
        -------
        reward : 扣除手续费与滑点后的净收益。
        done : 是否到达 horizon 末尾。
        info : 诊断信息（含 ``reject_event``）。

        Raises
        ------
        RuntimeError : 调用前未 ``reset()``。
        ValueError : action ∉ {0, 1, 2}。
        """
        if self._horizon is None:
            raise RuntimeError("step() 调用前必须先 reset()")
        if self._t >= len(self._horizon.execution_books):
            # 防御性: 调用方应在 done=True 后停止；继续 step 视为编程错误而非静默错误。
            raise RuntimeError(
                f"step() 已超过 horizon 长度 {len(self._horizon.execution_books)}; "
                "请先调用 reset() 重新开始 horizon。"
            )
        target_position = self._action_to_position(action)
        result = self._apply_execution(self._position, target_position, self._t)
        rows = self._rows_for(self._t)
        # reward = position * (markout - exec) - cost。
        # 注意: 即使 rejected，filled_position 仍 == prev_position，因此 reward 只算持仓收益。
        p_exec = float(self._horizon.prices[rows.execution_row])
        p_markout = float(self._horizon.prices[rows.markout_row])
        position_after = result.filled_position
        position_pnl = position_after * (p_markout - p_exec)
        reward = position_pnl - result.cost

        self._position = position_after
        self._nav += reward
        info = StepInfo(
            fee=result.fee,
            slippage=result.slippage,
            fill_price=result.fill_price,
            filled_position=result.filled_position,
            nav=self._nav,
            execution_row=rows.execution_row,
            markout_row=rows.markout_row,
            rejected=result.rejected,
            reject_event=(
                {
                    "step": self._t,
                    "prev_position": result.prev_position,
                    "target_position": result.target_position,
                    "reason": result.reject_reason,
                }
                if result.rejected
                else None
            ),
        )
        # done 判断: 当下一步会越过 horizon 长度时结束。
        # horizon 长度 = len(execution_books)；step 结束意味着这一步已经处理完。
        self._t += 1
        done = self._t >= len(self._horizon.execution_books)
        return reward, done, info

    def replay(self, actions: Sequence[int]) -> tuple[List[float], List[StepInfo]]:
        """一次性 replay 整段动作序列。

        实现注意: 必须等价于反复调用 ``step``；DP teacher 与 student replay
        都不允许走捷径计算 reward，否则会破坏 teacher / student 可比性。
        当 ``done`` 触发时立即终止，不执行多余的 actions。
        """
        rewards: List[float] = []
        infos: List[StepInfo] = []
        for action in actions:
            r, done, info = self.step(action)
            rewards.append(r)
            infos.append(info)
            if done:
                break
        return rewards, infos

    # ---------- 内部辅助 ----------

    def _action_to_position(self, action: int) -> int:
        """``{0, 1, 2}`` → ``{-m, 0, +m}`` 映射。"""
        if action not in (0, 1, 2):
            raise ValueError(f"非法 action: {action!r}; 仅允许 0/1/2")
        return self.max_position * (action - 1)

    def _rows_for(self, t: int) -> AlignmentRows:
        return self.reward_alignment.rows(t)

    def _apply_execution(
        self, prev_position: int, target_position: int, t: int
    ) -> ExecutionResult:
        """组合 cost_model 与 alignment，由 step/replay 复用。

        ``execution_books[t]`` 的语义已经在 ``HorizonBuilder`` 里按 reward_alignment
        对齐到对应行号（``paper_formula`` → 第 ``t`` 行；``next_row_execution`` →
        第 ``t+1`` 行盘口），因此本方法只需直接索引 ``t``，不需要再加 offset。
        """
        assert self._horizon is not None
        book = self._horizon.execution_books[t]
        return self.cost_model.execute(
            prev_position=prev_position,
            target_position=target_position,
            execution_book=book,
        )
