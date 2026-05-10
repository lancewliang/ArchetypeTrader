from __future__ import annotations

import numpy as np
import pytest

from src.utils import ActionExecutionCalculator


def test_action_execution_calculates_net_returns_with_fees() -> None:
    prices = np.array([[[100.0], [110.0], [99.0]]])
    actions = np.array([[2, 0, 1]])

    result = ActionExecutionCalculator(fee_rate=0.001).execute(prices, actions)

    np.testing.assert_allclose(result.gross_returns, [0.2])
    np.testing.assert_allclose(result.turnover, [4.0])
    np.testing.assert_allclose(result.fees, [0.004])
    np.testing.assert_allclose(result.returns, [0.196])


def test_action_execution_returns_nan_when_prices_are_missing() -> None:
    actions = np.array([[2, 1, 0], [1, 1, 1]])

    result = ActionExecutionCalculator(fee_rate=0.001).execute(None, actions)

    assert result.returns.shape == (2,)
    assert np.isnan(result.returns).all()
    assert np.isnan(result.gross_returns).all()
    assert np.isnan(result.fees).all()
    assert np.isnan(result.turnover).all()


def test_action_to_position_mapping_is_stable() -> None:
    positions = ActionExecutionCalculator.actions_to_positions(np.array([[0, 1, 2]]))

    np.testing.assert_allclose(positions, [[-1.0, 0.0, 1.0]])


def test_action_execution_rejects_sample_count_mismatch() -> None:
    prices = np.array([[100.0, 101.0]])
    actions = np.array([[2, 1], [0, 1]])

    with pytest.raises(ValueError, match="same sample count"):
        ActionExecutionCalculator().execute(prices, actions)
