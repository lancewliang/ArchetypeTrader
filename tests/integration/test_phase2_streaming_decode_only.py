"""Phase II streaming decode only 集成测试。"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.phase1_frozen_policy import DecodeStepOutput
from src.trading.horizon_env import HorizonEnv
from tests.phase2_test_utils import make_config, make_dataset, make_trading_env


class StreamingOnlyPolicy:
    def __init__(self) -> None:
        self.decode_step_calls = 0

    def reset(self, code_id: int) -> None:
        self.code_id = code_id

    def decode(self, *args, **kwargs):
        raise AssertionError("HorizonEnv must not call batch decode()")

    def decode_step(self, state_t):
        self.decode_step_calls += 1
        return DecodeStepOutput(
            action_logits=np.array([0.0, 1.0, 0.0]),
            action=1,
            recurrent_state=None,
        )


class TestPhase2StreamingDecodeOnly:

    @pytest.mark.integration
    def test_decode_not_called(self, tmp_path):
        """mock decode()，若被调用则测试失败。"""
        config = make_config(tmp_path, horizon=4)
        dataset = make_dataset(config, count=1)
        policy = StreamingOnlyPolicy()
        env = HorizonEnv(0, dataset, policy, make_trading_env(), config, [0])
        env.reset()
        env.step(0)

    @pytest.mark.integration
    def test_decode_step_called_h_times(self, tmp_path):
        """decode_step() 被调用 h 次。"""
        config = make_config(tmp_path, horizon=4)
        dataset = make_dataset(config, count=1)
        policy = StreamingOnlyPolicy()
        env = HorizonEnv(0, dataset, policy, make_trading_env(), config, [0])
        env.reset()
        env.step(0)
        assert policy.decode_step_calls == config.horizon
