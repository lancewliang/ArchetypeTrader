"""VQ Codebook 单元测试"""

import torch
import pytest

from src.phase1.codebook import VQCodebook


class TestVQCodebookInit:
    """码本初始化验证"""

    def test_default_params(self):
        cb = VQCodebook()
        assert cb.num_codes == 10
        assert cb.code_dim == 16
        assert cb.embeddings.weight.shape == (10, 16)

    def test_custom_params(self):
        cb = VQCodebook(num_codes=5, code_dim=8)
        assert cb.num_codes == 5
        assert cb.code_dim == 8
        assert cb.embeddings.weight.shape == (5, 8)


class TestVQCodebookQuantize:
    """quantize() 方法测试"""

    def test_output_shapes(self):
        cb = VQCodebook(num_codes=10, code_dim=16)
        z_e = torch.randn(4, 16)
        z_q_st, indices, commitment_loss = cb.quantize(z_e)

        assert z_q_st.shape == (4, 16)
        assert indices.shape == (4,)
        assert commitment_loss.dim() == 0  # scalar

    def test_indices_in_range(self):
        cb = VQCodebook(num_codes=10, code_dim=16)
        z_e = torch.randn(32, 16)
        _, indices, _ = cb.quantize(z_e)

        assert (indices >= 0).all()
        assert (indices < 10).all()

    def test_nearest_neighbor_correctness(self):
        """选中的索引应是真正的最近邻"""
        cb = VQCodebook(num_codes=10, code_dim=16)
        z_e = torch.randn(8, 16)
        _, indices, _ = cb.quantize(z_e)

        codebook = cb.embeddings.weight.detach()
        for i in range(z_e.shape[0]):
            dists = torch.sum((z_e[i] - codebook) ** 2, dim=1)
            expected_idx = torch.argmin(dists)
            assert indices[i] == expected_idx

    def test_commitment_loss_nonnegative(self):
        cb = VQCodebook(num_codes=10, code_dim=16)
        z_e = torch.randn(8, 16)
        _, _, commitment_loss = cb.quantize(z_e)
        assert commitment_loss.item() >= 0.0

    def test_straight_through_gradient(self):
        """z_q_st 应该让梯度流过 z_e，而不是流过码本"""
        cb = VQCodebook(num_codes=10, code_dim=16)
        z_e = torch.randn(4, 16, requires_grad=True)
        z_q_st, _, _ = cb.quantize(z_e)

        # 反向传播
        loss = z_q_st.sum()
        loss.backward()

        # z_e 应该有梯度（straight-through）
        assert z_e.grad is not None
        # 梯度应该全为 1（因为 z_q_st = z_e + (z_q - z_e).detach()，对 z_e 求导 = 1）
        assert torch.allclose(z_e.grad, torch.ones_like(z_e.grad))

    def test_commitment_loss_does_not_backprop_to_z_e(self):
        """commitment_loss = ||z_e.detach() - z_q||²，不应对 z_e 产生梯度"""
        cb = VQCodebook(num_codes=10, code_dim=16)
        z_e = torch.randn(4, 16, requires_grad=True)
        _, _, commitment_loss = cb.quantize(z_e)

        commitment_loss.backward()
        # commitment_loss 使用了 z_e.detach()，所以 z_e 不应有梯度
        assert z_e.grad is None or torch.allclose(z_e.grad, torch.zeros_like(z_e.grad))

    def test_z_q_st_values_match_codebook(self):
        """z_q_st 的前向值应等于码本中对应向量"""
        cb = VQCodebook(num_codes=10, code_dim=16)
        z_e = torch.randn(4, 16)
        z_q_st, indices, _ = cb.quantize(z_e)

        for i in range(4):
            expected = cb.embeddings.weight[indices[i]].detach()
            assert torch.allclose(z_q_st[i].detach(), expected)


from src.phase1.vq_encoder import VQEncoder


class TestVQEncoderInit:
    """VQ Encoder 初始化验证"""

    def test_default_params(self):
        enc = VQEncoder(state_dim=45)
        assert enc.state_dim == 45
        assert enc.action_dim == 3
        assert enc.hidden_dim == 128
        assert enc.latent_dim == 16
        assert enc.lstm.input_size == 45 + 3 + 1  # state + one-hot action + reward
        assert enc.lstm.hidden_size == 128
        assert enc.projection.in_features == 128
        assert enc.projection.out_features == 16

    def test_custom_params(self):
        enc = VQEncoder(state_dim=10, action_dim=5, hidden_dim=64, latent_dim=8)
        assert enc.lstm.input_size == 10 + 5 + 1
        assert enc.lstm.hidden_size == 64
        assert enc.projection.out_features == 8


class TestVQEncoderForward:
    """VQ Encoder forward() 方法测试"""

    def test_output_shape(self):
        enc = VQEncoder(state_dim=45)
        s = torch.randn(4, 72, 45)
        a = torch.randint(0, 3, (4, 72))
        r = torch.randn(4, 72)
        z_e = enc(s, a, r)
        assert z_e.shape == (4, 16)

    def test_output_shape_single_sample(self):
        enc = VQEncoder(state_dim=45)
        s = torch.randn(1, 72, 45)
        a = torch.randint(0, 3, (1, 72))
        r = torch.randn(1, 72)
        z_e = enc(s, a, r)
        assert z_e.shape == (1, 16)

    def test_output_shape_short_horizon(self):
        """不同 horizon 长度也应正常工作"""
        enc = VQEncoder(state_dim=45)
        s = torch.randn(2, 10, 45)
        a = torch.randint(0, 3, (2, 10))
        r = torch.randn(2, 10)
        z_e = enc(s, a, r)
        assert z_e.shape == (2, 16)

    def test_gradient_flows(self):
        """z_e 应支持反向传播"""
        enc = VQEncoder(state_dim=45)
        s = torch.randn(2, 72, 45)
        a = torch.randint(0, 3, (2, 72))
        r = torch.randn(2, 72)
        z_e = enc(s, a, r)
        loss = z_e.sum()
        loss.backward()
        # 检查 LSTM 和 projection 参数都有梯度
        assert enc.projection.weight.grad is not None
        for name, param in enc.lstm.named_parameters():
            assert param.grad is not None, f"LSTM param {name} has no gradient"

    def test_different_inputs_different_outputs(self):
        """不同输入应产生不同嵌入"""
        enc = VQEncoder(state_dim=45)
        s1 = torch.randn(1, 72, 45)
        s2 = torch.randn(1, 72, 45)
        a = torch.randint(0, 3, (1, 72))
        r = torch.randn(1, 72)
        z1 = enc(s1, a, r)
        z2 = enc(s2, a, r)
        assert not torch.allclose(z1, z2)


from src.phase1.vq_decoder import VQDecoder


class TestVQDecoderInit:
    """VQ Decoder 初始化验证"""

    def test_default_params(self):
        dec = VQDecoder(state_dim=45)
        assert dec.state_dim == 45
        assert dec.code_dim == 16
        assert dec.hidden_dim == 128
        assert dec.action_dim == 3

    def test_custom_params(self):
        dec = VQDecoder(state_dim=10, code_dim=8, hidden_dim=64, action_dim=5)
        assert dec.state_dim == 10
        assert dec.code_dim == 8
        assert dec.hidden_dim == 64
        assert dec.action_dim == 5

    def test_mlp_layer_dims(self):
        dec = VQDecoder(state_dim=45, code_dim=16, hidden_dim=128, action_dim=3)
        # First linear: (state_dim + code_dim) → hidden_dim
        assert dec.mlp[0].in_features == 45 + 16
        assert dec.mlp[0].out_features == 128
        # Second linear: hidden_dim → action_dim
        assert dec.mlp[2].in_features == 128
        assert dec.mlp[2].out_features == 3


class TestVQDecoderForward:
    """VQ Decoder forward() 方法测试"""

    def test_output_shape(self):
        dec = VQDecoder(state_dim=45)
        states = torch.randn(4, 72, 45)
        z_q = torch.randn(4, 16)
        logits = dec(states, z_q)
        assert logits.shape == (4, 72, 3)

    def test_output_shape_single_sample(self):
        dec = VQDecoder(state_dim=45)
        states = torch.randn(1, 72, 45)
        z_q = torch.randn(1, 16)
        logits = dec(states, z_q)
        assert logits.shape == (1, 72, 3)

    def test_output_shape_short_horizon(self):
        """不同 horizon 长度也应正常工作"""
        dec = VQDecoder(state_dim=45)
        states = torch.randn(2, 10, 45)
        z_q = torch.randn(2, 16)
        logits = dec(states, z_q)
        assert logits.shape == (2, 10, 3)

    def test_output_shape_custom_dims(self):
        dec = VQDecoder(state_dim=10, code_dim=8, hidden_dim=64, action_dim=5)
        states = torch.randn(3, 20, 10)
        z_q = torch.randn(3, 8)
        logits = dec(states, z_q)
        assert logits.shape == (3, 20, 5)

    def test_argmax_produces_valid_actions(self):
        """argmax 后值域应为 {0, 1, 2}"""
        dec = VQDecoder(state_dim=45, action_dim=3)
        states = torch.randn(8, 72, 45)
        z_q = torch.randn(8, 16)
        logits = dec(states, z_q)
        actions = torch.argmax(logits, dim=-1)
        assert actions.shape == (8, 72)
        assert (actions >= 0).all()
        assert (actions < 3).all()

    def test_gradient_flows(self):
        """logits 应支持反向传播到 decoder 参数"""
        dec = VQDecoder(state_dim=45)
        states = torch.randn(2, 72, 45)
        z_q = torch.randn(2, 16, requires_grad=True)
        logits = dec(states, z_q)
        loss = logits.sum()
        loss.backward()
        # z_q 应有梯度
        assert z_q.grad is not None
        # MLP 参数应有梯度
        assert dec.mlp[0].weight.grad is not None
        assert dec.mlp[2].weight.grad is not None

    def test_different_z_q_different_outputs(self):
        """不同 z_q 应产生不同输出"""
        dec = VQDecoder(state_dim=45)
        states = torch.randn(1, 72, 45)
        z_q1 = torch.randn(1, 16)
        z_q2 = torch.randn(1, 16)
        logits1 = dec(states, z_q1)
        logits2 = dec(states, z_q2)
        assert not torch.allclose(logits1, logits2)

    def test_same_z_q_broadcast_across_time(self):
        """同一 z_q 应在所有时间步共享"""
        dec = VQDecoder(state_dim=45)
        # 使用相同状态在每个时间步，验证 z_q 广播一致
        single_state = torch.randn(1, 1, 45)
        states = single_state.expand(1, 5, 45)
        z_q = torch.randn(1, 16)
        logits = dec(states, z_q)
        # 所有时间步输入相同 → 输出应相同
        for t in range(1, 5):
            assert torch.allclose(logits[0, 0], logits[0, t])


# ============================================================
# Property-Based Tests (hypothesis)
# ============================================================

from hypothesis import given, settings
from hypothesis import strategies as st


# Feature: archetype-trader, Property 12: VQ 维度不变量
class TestPropVQDimensionInvariant:
    """Property 12: VQ 维度不变量
    z_e 维度 16，码本 K=10 × 16

    **Validates: Requirements 4.2, 4.3**
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        seq_len=st.integers(min_value=1, max_value=72),
    )
    @settings(max_examples=100)
    def test_prop_vq_dimension_invariant(self, batch_size: int, seq_len: int):
        encoder = VQEncoder(state_dim=45, action_dim=3, hidden_dim=128, latent_dim=16)
        codebook = VQCodebook(num_codes=10, code_dim=16)

        s_demo = torch.randn(batch_size, seq_len, 45)
        a_demo = torch.randint(0, 3, (batch_size, seq_len))
        r_demo = torch.randn(batch_size, seq_len)

        with torch.no_grad():
            z_e = encoder(s_demo, a_demo, r_demo)

        # z_e 维度应为 (batch, 16)
        assert z_e.shape == (batch_size, 16)
        # 码本应为 (10, 16)
        assert codebook.embeddings.weight.shape == (10, 16)


# Feature: archetype-trader, Property 13: 最近邻量化正确性
class TestPropNearestNeighborQuantization:
    """Property 13: 最近邻量化正确性
    选中索引 k 满足 ||z_e - e_k|| ≤ ||z_e - e_j|| ∀j

    **Validates: Requirements 4.4**
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_prop_nearest_neighbor_correctness(self, batch_size: int, data):
        codebook = VQCodebook(num_codes=10, code_dim=16)
        z_e = torch.randn(batch_size, 16)

        with torch.no_grad():
            _, indices, _ = codebook.quantize(z_e)

        cb_weight = codebook.embeddings.weight.detach()
        for i in range(batch_size):
            # 计算到所有码本向量的距离
            dists = torch.sum((z_e[i] - cb_weight) ** 2, dim=1)
            selected_dist = dists[indices[i]]
            # 选中索引的距离应 ≤ 所有其他索引的距离
            assert torch.all(selected_dist <= dists + 1e-6), (
                f"Sample {i}: selected index {indices[i]} dist={selected_dist.item():.6f} "
                f"but min dist={dists.min().item():.6f} at index {dists.argmin().item()}"
            )


# Feature: archetype-trader, Property 14: 解码器输出有效动作
class TestPropDecoderValidActions:
    """Property 14: 解码器输出有效动作
    argmax 后值域 {0, 1, 2}

    **Validates: Requirements 4.5**
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        seq_len=st.integers(min_value=1, max_value=72),
    )
    @settings(max_examples=100)
    def test_prop_decoder_output_valid_actions(self, batch_size: int, seq_len: int):
        decoder = VQDecoder(state_dim=45, code_dim=16, hidden_dim=128, action_dim=3)

        states = torch.randn(batch_size, seq_len, 45)
        z_q = torch.randn(batch_size, 16)

        with torch.no_grad():
            logits = decoder(states, z_q)

        # argmax 得到动作
        actions = torch.argmax(logits, dim=-1)

        # 动作值域应为 {0, 1, 2}
        assert actions.shape == (batch_size, seq_len)
        assert torch.all(actions >= 0)
        assert torch.all(actions <= 2)


# Feature: archetype-trader, Property 15: VQ 损失函数正确性
class TestPropVQLossCorrectness:
    """Property 15: VQ 损失函数正确性
    L = L_rec + ||sg[z_e] - z_q||² + 0.25 × ||z_e - sg[z_q]||²

    **Validates: Requirements 4.6**
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=32),
    )
    @settings(max_examples=100)
    def test_prop_vq_loss_decomposition(self, batch_size: int):
        codebook = VQCodebook(num_codes=10, code_dim=16)
        z_e = torch.randn(batch_size, 16)

        z_q_st, indices, commitment_loss = codebook.quantize(z_e)

        # commitment_loss = ||sg[z_e] - z_q||² (mean)
        z_q = codebook.embeddings(indices)
        expected_commitment = torch.mean((z_e.detach() - z_q) ** 2)
        assert torch.allclose(commitment_loss, expected_commitment, atol=1e-5), (
            f"commitment_loss={commitment_loss.item():.6f} != "
            f"expected={expected_commitment.item():.6f}"
        )

        # β₀ term: ||z_e - sg[z_q]||²
        beta0 = 0.25
        encoder_loss = torch.mean((z_e - z_q.detach()) ** 2)

        # 模拟一个简单的重建损失
        dummy_rec_loss = torch.tensor(0.5)

        # 总损失分解验证
        total_loss = dummy_rec_loss + commitment_loss + beta0 * encoder_loss
        expected_total = dummy_rec_loss + expected_commitment + beta0 * encoder_loss
        assert torch.allclose(total_loss, expected_total, atol=1e-5), (
            f"total_loss={total_loss.item():.6f} != "
            f"expected_total={expected_total.item():.6f}"
        )
