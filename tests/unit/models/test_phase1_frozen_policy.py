"""Phase I frozen policy 单元测试。"""
from __future__ import annotations

import pytest
import torch

from src.models.phase1_frozen_policy import Phase1FrozenPolicy
from src.models.vq_archetype import ArchetypeDecoder


def _policy() -> Phase1FrozenPolicy:
    decoder = ArchetypeDecoder(feature_dim=3, code_dim=4, hidden_dim=8)
    codebook = torch.randn(3, 4) * 0.05
    return Phase1FrozenPolicy(decoder, codebook, device="cpu")


class TestPhase1FrozenPolicy:

    def test_decoder_parameters_frozen(self):
        """decoder 参数被冻结。"""
        policy = _policy()
        assert all(not p.requires_grad for p in policy.decoder.parameters())

    def test_decode_step_outputs_action_logits(self):
        """decode_step() 可逐步输出 action logits。"""
        policy = _policy()
        policy.reset(code_id=1)
        out = policy.decode_step(torch.zeros(3))
        assert out.action_logits.shape == (3,)
        assert out.action in (0, 1, 2)
        assert out.recurrent_state is not None

    def test_decode_step_requires_reset(self):
        """decode_step() 调用前必须 reset()。"""
        policy = _policy()
        with pytest.raises(RuntimeError):
            policy.decode_step(torch.zeros(3))

    def test_modifying_future_state_does_not_change_past_logits(self):
        """批量诊断 decode 保持因果性: 修改未来 state 不影响过去 logits。"""
        policy = _policy()
        states = torch.zeros(5, 3)
        changed = states.clone()
        changed[3:] = 100.0

        _, logits_a = policy.decode(states, code_id=0)
        _, logits_b = policy.decode(changed, code_id=0)
        assert torch.allclose(logits_a[:3], logits_b[:3], atol=1e-6)

    def test_rejects_bidirectional_decoder(self):
        """双向 LSTM decoder 被拒绝。"""
        decoder = ArchetypeDecoder(feature_dim=3, code_dim=4, hidden_dim=8)
        decoder.lstm = torch.nn.LSTM(
            input_size=12,
            hidden_size=8,
            batch_first=True,
            bidirectional=True,
        )
        with pytest.raises(ValueError):
            Phase1FrozenPolicy(decoder, torch.randn(3, 4), device="cpu")
