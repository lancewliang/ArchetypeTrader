"""Phase II online action throttle utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.config.phase2_config import OnlineActionThrottleConfig


@dataclass
class ActionThrottleDecision:
    """Result of applying online action throttling."""
    action: int
    original_action: int
    triggered: bool = False
    reason: Optional[str] = None
    cooldown_remaining: int = 0


class Phase2OnlineActionThrottle:
    """Stateful selector-action throttle for backtest/live inference."""

    def __init__(
        self,
        config: OnlineActionThrottleConfig,
        flat_code: int = 0,
    ) -> None:
        self.config = config
        self.flat_code = int(flat_code)
        self._last_action: Optional[int] = None
        self._recent_switches: List[int] = []
        self._cooldown_remaining = 0

    def apply(
        self,
        action: int,
        confidence: Optional[float] = None,
        position_delta: Optional[int] = None,
    ) -> ActionThrottleDecision:
        """Apply confidence, switch-rate, cooldown and position-change limits."""
        original = int(action)
        action = int(action)

        if (
            confidence is not None
            and confidence < self.config.min_confidence_for_non_flat_action
            and action != self.flat_code
        ):
            self._remember(self.flat_code)
            return ActionThrottleDecision(
                action=self.flat_code,
                original_action=original,
                triggered=True,
                reason="low_confidence",
                cooldown_remaining=self._cooldown_remaining,
            )

        if self._cooldown_remaining > 0 and self._last_action is not None:
            self._cooldown_remaining -= 1
            kept = self._last_action
            self._remember(kept)
            return ActionThrottleDecision(
                action=kept,
                original_action=original,
                triggered=True,
                reason="cooldown",
                cooldown_remaining=self._cooldown_remaining,
            )

        if (
            self.config.max_position_change_per_horizon is not None
            and position_delta is not None
            and abs(position_delta) > self.config.max_position_change_per_horizon
        ):
            kept = self._last_action if self._last_action is not None else self.flat_code
            self._remember(kept)
            return ActionThrottleDecision(
                action=kept,
                original_action=original,
                triggered=True,
                reason="max_position_change",
                cooldown_remaining=self._cooldown_remaining,
            )

        switched = self._last_action is not None and action != self._last_action
        if switched:
            self._recent_switches.append(1)
        else:
            self._recent_switches.append(0)
        self._recent_switches = self._recent_switches[-max(self.config.switch_window_n, 1):]

        if sum(self._recent_switches) > self.config.max_archetype_switches_per_n_horizons:
            self._cooldown_remaining = max(self.config.cooldown_after_large_turnover, 0)
            kept = self._last_action if self._last_action is not None else action
            self._remember(kept)
            return ActionThrottleDecision(
                action=kept,
                original_action=original,
                triggered=True,
                reason="switch_frequency",
                cooldown_remaining=self._cooldown_remaining,
            )

        self._remember(action)
        return ActionThrottleDecision(
            action=action,
            original_action=original,
            triggered=False,
            cooldown_remaining=self._cooldown_remaining,
        )

    def _remember(self, action: int) -> None:
        self._last_action = int(action)

