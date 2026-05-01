"""Phase II online action throttle 单元测试。"""
from __future__ import annotations

from src.config.phase2_config import OnlineActionThrottleConfig
from src.evaluation.phase2_online_action_throttle import Phase2OnlineActionThrottle


class TestPhase2OnlineActionThrottle:

    def test_low_confidence_forces_flat(self):
        """低置信度强制 flat_only。"""
        throttle = Phase2OnlineActionThrottle(
            OnlineActionThrottleConfig(min_confidence_for_non_flat_action=0.9),
            flat_code=0,
        )
        decision = throttle.apply(2, confidence=0.1)
        assert decision.triggered is True
        assert decision.reason == "low_confidence"
        assert decision.action == 0

    def test_switch_frequency_triggers_cooldown(self):
        """archetype 切换频率超阈时触发 cooldown。"""
        throttle = Phase2OnlineActionThrottle(
            OnlineActionThrottleConfig(
                max_archetype_switches_per_n_horizons=1,
                switch_window_n=3,
                cooldown_after_large_turnover=2,
            )
        )
        throttle.apply(0)
        throttle.apply(1)
        decision = throttle.apply(2)
        assert decision.triggered is True
        assert decision.reason == "switch_frequency"
        assert decision.cooldown_remaining == 2

    def test_max_position_change_per_horizon(self):
        """max_position_change_per_horizon 生效。"""
        throttle = Phase2OnlineActionThrottle(
            OnlineActionThrottleConfig(max_position_change_per_horizon=1),
            flat_code=0,
        )
        throttle.apply(1)
        decision = throttle.apply(2, position_delta=3)
        assert decision.triggered is True
        assert decision.reason == "max_position_change"
        assert decision.action == 1
