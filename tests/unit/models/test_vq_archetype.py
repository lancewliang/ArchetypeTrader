"""``VQArchetypeModel`` 单元测试 (含因果约束)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.phase1.config import (
    CodebookConfig,
    CodebookHealthConfig,
    EncoderInputConfig,
    ModelConfig,
)
from src.phase1.models.vq_archetype import VQArchetypeModel


def _make_model(feature_dim=2, num_codes=4):
    enc = EncoderInputConfig(state_adapter_dim=8, action_embedding_dim=4, reward_embedding_dim=4, fusion_dim=16)
    health = CodebookHealthConfig(usage_regularization_weight=0.0, dead_code_restart=False)
    cb = CodebookConfig(init_method="random_normal", update_method="ema", health=health)
    model = ModelConfig(hidden_dim=16, code_dim=8, num_codes=num_codes, encoder_input=enc, codebook=cb)
    return VQArchetypeModel(feature_dim=feature_dim, config=model)


def test_forward_logits_shape():
    model = _make_model()
    states = torch.randn(2, 6, 2)
    actions = torch.zeros(2, 6, dtype=torch.long)
    rewards = torch.zeros(2, 6)
    out = model(states, actions, rewards)
    assert out.action_logits.shape == (2, 6, 3)
    assert out.code_id.shape == (2,)


def test_forward_pretrain_skips_quantizer_and_uses_z_e_directly():
    model = _make_model()

    def _raise_if_called(_z_e):
        raise AssertionError("quantizer should not run during Phase A pretrain")

    model.quantizer.quantize = _raise_if_called
    states = torch.randn(2, 6, 2)
    actions = torch.zeros(2, 6, dtype=torch.long)
    rewards = torch.zeros(2, 6)
    out = model.forward_pretrain(states, actions, rewards)
    assert out.action_logits.shape == (2, 6, 3)
    assert out.code_id is None
    assert torch.allclose(out.z_q, out.z_e)
    assert torch.allclose(out.z_q_no_grad, out.z_e.detach())


def test_decoder_is_unidirectional():
    """硬约束: decoder.lstm.bidirectional == False。"""
    model = _make_model()
    assert model.decoder.lstm.bidirectional is False


def test_modifying_future_states_does_not_change_past_logits():
    """因果性硬测试: 修改 s_{τ+1:} 不应改变 logits[:, :τ+1, :]。"""
    model = _make_model()
    model.eval()
    states = torch.randn(1, 6, 2)
    actions = torch.zeros(1, 6, dtype=torch.long)
    rewards = torch.zeros(1, 6)
    with torch.no_grad():
        out_a = model(states, actions, rewards).action_logits
    states_b = states.clone()
    states_b[0, 4:] = states_b[0, 4:] + 999.0  # 修改未来 states
    with torch.no_grad():
        out_b = model(states_b, actions, rewards).action_logits
    # 注意: encoder 是基于全段 states 的，这会影响 z_q；
    # 因此前段 logits 也可能因 z_q 变化而变化。
    # 真正的因果性测试只针对 decoder：固定 z_q，只改 future states。
    fused = model.input_adapter(states, actions, rewards)
    z_e = model.encoder(fused)
    q = model.quantizer.quantize(z_e)
    with torch.no_grad():
        logits_a = model.decoder(states, q.z_q)
        logits_b = model.decoder(states_b, q.z_q)
    # 第 0 步只看 s_0 + z_q，应当完全相同
    assert torch.allclose(logits_a[:, 0, :], logits_b[:, 0, :])
    assert torch.allclose(logits_a[:, 3, :], logits_b[:, 3, :])
    # 第 4 步及之后会变化
    assert not torch.allclose(logits_a[:, 4, :], logits_b[:, 4, :])


def test_encode_returns_code_id_and_z_e():
    model = _make_model()
    states = torch.randn(2, 6, 2)
    actions = torch.zeros(2, 6, dtype=torch.long)
    rewards = torch.zeros(2, 6)
    code_id, z_e = model.encode(states, actions, rewards)
    assert code_id.shape == (2,)
    assert z_e.shape == (2, 8)


def test_decode_uses_codebook_lookup():
    model = _make_model()
    states = torch.randn(2, 6, 2)
    code_id = torch.tensor([0, 1])
    logits = model.decode(states, code_id)
    assert logits.shape == (2, 6, 3)
