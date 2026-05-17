import math

from src.phase1.evaluators.phase1_validation_layers.layer3_oracle_profitability import (
    compute_downside_control,
)


def test_downside_control_zero_teacher_zero_decoded_drawdown_passes_as_zero() -> None:
    assert compute_downside_control(decoded_drawdown=0.0, dp_drawdown=0.0) == 0.0


def test_downside_control_zero_teacher_positive_decoded_drawdown_is_infinite() -> None:
    assert math.isinf(
        compute_downside_control(decoded_drawdown=1.0, dp_drawdown=0.0)
    )


def test_downside_control_uses_ratio_when_teacher_drawdown_is_positive() -> None:
    assert compute_downside_control(decoded_drawdown=3.0, dp_drawdown=2.0) == 1.5
