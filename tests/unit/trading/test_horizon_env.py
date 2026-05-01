"""HorizonEnv 单元测试。

测试用例:
- reset() 返回第一个 horizon 的 s^sel。
- step(action) 会执行完整 horizon 并返回 r^sel。
- prev_terminal_position 正确继承到下一个 horizon。
- decode_step() 被调用 h 次，而不是批量 decode()。
- position_continuity=false 时每个 horizon 从 flat reset。
- gap <= threshold 且 mode=carry 时继承仓位。
- gap > threshold 且 mode=force_flatten 时仓位归零。
- gap > threshold 且 mode=warmup_only 时 warm-up 行为正确。
- 风险触发时支持 mid-horizon flatten。
"""
import pytest


class TestHorizonEnv:

    def test_reset_returns_selector_state(self):
        """reset() 返回第一个 horizon 的 s^sel。"""
        pass

    def test_step_executes_full_horizon(self):
        """step(action) 执行完整 horizon 并返回 r^sel。"""
        pass

    def test_prev_terminal_position_inherited(self):
        """prev_terminal_position 正确继承到下一个 horizon。"""
        pass

    def test_decode_step_called_h_times(self):
        """decode_step() 被调用 h 次，而不是批量 decode()。"""
        pass

    def test_no_position_continuity_flat_reset(self):
        """position_continuity=false 时每个 horizon 从 flat reset。"""
        pass

    def test_gap_carry_mode(self):
        """gap <= threshold 且 mode=carry 时继承仓位。"""
        pass

    def test_gap_force_flatten_mode(self):
        """gap > threshold 且 mode=force_flatten 时仓位归零。"""
        pass

    def test_gap_warmup_only_mode(self):
        """gap > threshold 且 mode=warmup_only 时 warm-up 行为正确。"""
        pass

    def test_mid_horizon_flatten_on_risk(self):
        """风险触发时支持 mid-horizon flatten。"""
        pass
