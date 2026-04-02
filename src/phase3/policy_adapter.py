"""Policy Adapter — 策略适配器

# 论文 Eq. 6: 最终动作计算
# a_final = a_base  if a_base ≠ a_base_{t-1} or a_ref = 0
# a_final = 0       if a_ref = -1
# a_final = 2       if a_ref = 1
# 约束: 每个 horizon 最多一次调整
#
# PolicyAdapter 在 Refinement Agent 输出调整信号后，
# 结合基础动作和约束条件计算最终交易动作。
#
# 设计: compute_final_action 为纯函数，无副作用。
# 调整状态 (has_adjusted) 由调用方显式管理。
"""

from typing import Tuple


class PolicyAdapter:
    """策略适配器，按论文 Eq. 6 计算最终交易动作（纯函数设计）。

    # Eq. 6: 最终动作计算
    # 规则优先级:
    #   1. 若本 horizon 已调整过 (has_adjusted=True) → a_final = a_base（忽略 a_ref）
    #   2. 若 a_base ≠ a_base_prev → a_final = a_base（基础动作变化，不精炼）
    #   3. 若 a_ref = 0 → a_final = a_base（无调整信号）
    #   4. 若 a_ref = -1 → a_final = 0（减仓至 short/flat），标记已调整
    #   5. 若 a_ref = 1 → a_final = 2（加仓至 long），标记已调整
    #
    # compute_final_action 为纯函数: 相同输入始终产生相同输出，无副作用。
    # 调整状态通过 has_adjusted 参数传入，通过返回值元组传出。
    """

    def __init__(self) -> None:
        pass

    def compute_final_action(
        self, a_base: int, a_base_prev: int, a_ref: int, has_adjusted: bool
    ) -> Tuple[int, bool]:
        """根据 Eq. 6 计算最终动作（纯函数，无副作用）。

        # Eq. 6: 最终动作计算
        # 输入: a_base (decoder 基础动作), a_base_prev (上一步基础动作),
        #        a_ref (调整信号), has_adjusted (当前 horizon 是否已调整)
        # 输出: (a_final, new_has_adjusted) 元组

        Args:
            a_base: 当前 decoder 输出的基础动作 ∈ {0, 1, 2}
            a_base_prev: 上一步的基础动作 ∈ {0, 1, 2}
            a_ref: refinement agent 的调整信号 ∈ {-1, 0, 1}
            has_adjusted: 当前 horizon 是否已执行过调整

        Returns:
            (a_final, new_has_adjusted) 元组:
                a_final: 最终交易动作 ∈ {0, 1, 2}
                new_has_adjusted: 更新后的调整状态
        """
        # Eq. 6, 约束: 每个 horizon 最多一次调整
        if has_adjusted:
            return (a_base, True)

        # Eq. 6: a_base ≠ a_base_prev → 基础动作已变化，直接使用
        if a_base != a_base_prev:
            return (a_base, has_adjusted)

        # Eq. 6: a_ref = 0 → 无调整
        if a_ref == 0:
            return (a_base, has_adjusted)

        # Eq. 6: a_ref = -1 → 减仓（a_final = 0, short/flat）
        if a_ref == -1:
            return (0, True)

        # Eq. 6: a_ref = 1 → 加仓（a_final = 2, long）
        if a_ref == 1:
            return (2, True)

        # 兜底: 未知 a_ref 值，返回 a_base
        return (a_base, has_adjusted)
