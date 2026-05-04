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

    def test_spawn_worker_policy_has_independent_streaming_state(self):
        """worker policy 共享冻结权重但不共享 recurrent runtime state。"""
        policy = _policy()
        worker_a = policy.spawn_worker_policy()
        worker_b = policy.spawn_worker_policy()

        assert worker_a is not worker_b
        assert worker_a.decoder is policy.decoder
        worker_a.reset(code_id=0)
        out_a = worker_a.decode_step(torch.zeros(3))

        worker_b.reset(code_id=1)
        out_b = worker_b.decode_step(torch.ones(3))

        assert out_a.recurrent_state is not worker_b._recurrent_state
        assert out_b.recurrent_state is worker_b._recurrent_state
        assert worker_a._code_id == 0
        assert worker_b._code_id == 1

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

    def test_load_accepts_prefixed_decoder_state_dict(self, tmp_path):
        """兼容旧 Phase I 导出的 decoder.* key。"""
        decoder = ArchetypeDecoder(feature_dim=3, code_dim=4, hidden_dim=8)
        prefixed = {
            f"decoder.{key}": value
            for key, value in decoder.state_dict().items()
        }
        decoder_path = tmp_path / "decoder.pt"
        codebook_path = tmp_path / "codebook.pt"
        torch.save(prefixed, decoder_path)
        torch.save(torch.randn(3, 4) * 0.05, codebook_path)

        policy = Phase1FrozenPolicy.load(decoder_path, codebook_path, device="cpu")

        policy.reset(0)
        assert policy.decode_step(torch.zeros(3)).action in (0, 1, 2)

    def test_load_accepts_quantizer_codebook_state_dict(self, tmp_path):
        """兼容旧 Phase I 导出的 quantizer.codebook key。"""
        decoder = ArchetypeDecoder(feature_dim=3, code_dim=4, hidden_dim=8)
        decoder_path = tmp_path / "decoder.pt"
        codebook_path = tmp_path / "codebook.pt"
        torch.save(decoder.state_dict(), decoder_path)
        torch.save({"quantizer.codebook": torch.randn(3, 4) * 0.05}, codebook_path)

        policy = Phase1FrozenPolicy.load(decoder_path, codebook_path, device="cpu")

        assert policy.num_codes == 3
