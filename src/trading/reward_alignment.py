"""Reward 时间对齐 (paper_formula / next_row_execution).

设计文档锚点: §4.4 (planners 表) 与 §5.3.1。

为什么独立成模块: DP teacher、env、student replay 必须共用同一套行号映射，
任何分歧都会导致老师/学生 reward 不可比。把映射集中到本文件并单元测试覆盖，
其他位置只能调用而不能自实现。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AlignmentMode = Literal["paper_formula", "next_row_execution"]


@dataclass(frozen=True)
class AlignmentRows:
    """单步行号映射结果。

    Attributes
    ----------
    decision_row: 决策时刻使用的状态/盘口行号。
    execution_row: 实际成交（fee + slippage 计算）使用的盘口行号。
    markout_row: 持仓收益结算使用的 mark price 行号。
    """
    decision_row: int
    execution_row: int
    markout_row: int


class RewardAlignment:
    """统一的行号映射对象。

    用法::

        align = RewardAlignment("paper_formula")
        rows = align.rows(t)  # decision=t, execution=t, markout=t+1

    实现注意:
    - 不引用 horizon 长度，调用方自己保证 ``markout_row`` 不越界。
    - 严禁在 mode 之外做隐式 fallback；非法 mode 直接抛 ``ValueError``。
    - 这是 DP teacher / env / student replay / boundary replay 的"唯一信源"，
      其他位置不允许自行计算行号偏移。
    """

    _VALID_MODES = ("paper_formula", "next_row_execution")

    def __init__(self, mode: AlignmentMode) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"非法 reward_alignment: {mode!r}; 仅支持 {self._VALID_MODES}"
            )
        self.mode: AlignmentMode = mode

    def rows(self, decision_offset: int) -> AlignmentRows:
        """根据决策偏移返回三类行号。

        Parameters
        ----------
        decision_offset : 在当前 horizon 内的决策时刻 ``t``，从 0 起。

        Returns
        -------
        AlignmentRows
            ``paper_formula`` 模式: ``decision=t, execution=t, markout=t+1``。
            ``next_row_execution`` 模式: ``decision=t, execution=t+1, markout=t+2``。

        Notes
        -----
        - ``paper_formula`` 严格对齐论文公式 ``r_t = P_t * (p_{t+1} - p_t) - O_t``：
          决策与成交都用第 ``t`` 行盘口/价格，用第 ``t+1`` 行 mark price 结算持仓收益。
        - ``next_row_execution`` 是更保守的在线仿真：成交整体后移一行，
          避免"用 ``t`` 行 features 再用 ``t`` 行 price 成交"的隐性泄漏；
          它的 reward 与论文公式不可直接比较。
        """
        if self.mode == "paper_formula":
            # 论文公式: r_t = P_t * (p_{t+1} - p_t) - O_t
            # 决策与成交都在第 t 行；用 t+1 行 mark 结算。
            return AlignmentRows(
                decision_row=decision_offset,
                execution_row=decision_offset,
                markout_row=decision_offset + 1,
            )
        # next_row_execution: 决策仍在 t；成交后移一行（更保守地避免行内泄漏）；markout 再后移一行。
        return AlignmentRows(
            decision_row=decision_offset,
            execution_row=decision_offset + 1,
            markout_row=decision_offset + 2,
        )

    def required_lookahead_rows(self) -> int:
        """每个 horizon 需要在数据末尾预留的额外行数。

        - ``paper_formula``: 1（用于 ``t = h - 1`` 的 markout）。
        - ``next_row_execution``: 2（execution 后移 1，markout 再后移 1）。

        ``HorizonBuilder`` 据此决定 ``prices`` 切片长度（``h`` 或 ``h+1`` 或 ``h+2``）。
        """
        return 1 if self.mode == "paper_formula" else 2
