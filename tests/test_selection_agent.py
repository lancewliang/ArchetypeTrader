"""Selection Agent 单元测试

测试 SelectionAgent 的前向传播、原型选择、保存/加载一致性和梯度流。
"""

import torch
import pytest
import tempfile
import os

from src.phase2.selection_agent import SelectionAgent


class TestSelectionAgentForwardShapes:
    """forward() 输出形状验证"""

    def test_output_shapes_default(self):
        """action_probs (batch, K), value (batch, 1)"""
        agent = SelectionAgent(state_dim=45, num_archetypes=10)
        state = torch.randn(4, 45)
        action_probs, value = agent(state)

        assert action_probs.shape == (4, 10)
        assert value.shape == (4, 1)

    def test_output_shapes_single_sample(self):
        agent = SelectionAgent(state_dim=45)
        state = torch.randn(1, 45)
        action_probs, value = agent(state)

        assert action_probs.shape == (1, 10)
        assert value.shape == (1, 1)

    def test_output_shapes_large_batch(self):
        agent = SelectionAgent(state_dim=45)
        state = torch.randn(64, 45)
        action_probs, value = agent(state)

        assert action_probs.shape == (64, 10)
        assert value.shape == (64, 1)

    def test_output_shapes_custom_params(self):
        agent = SelectionAgent(state_dim=20, num_archetypes=5)
        state = torch.randn(8, 20)
        action_probs, value = agent(state)

        assert action_probs.shape == (8, 5)
        assert value.shape == (8, 1)


class TestSelectionAgentProbabilityDistribution:
    """action_probs 是有效概率分布（非负，和为 1）"""

    def test_probs_nonnegative(self):
        agent = SelectionAgent(state_dim=45)
        state = torch.randn(8, 45)
        action_probs, _ = agent(state)

        assert (action_probs >= 0).all()

    def test_probs_sum_to_one(self):
        agent = SelectionAgent(state_dim=45)
        state = torch.randn(8, 45)
        action_probs, _ = agent(state)

        sums = action_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_probs_valid_with_extreme_input(self):
        """极端输入下概率仍应有效"""
        agent = SelectionAgent(state_dim=45)
        state = torch.randn(4, 45) * 100  # large magnitude
        action_probs, _ = agent(state)

        assert (action_probs >= 0).all()
        sums = action_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)


class TestSelectArchetype:
    """select_archetype() 推理方法测试"""

    def test_returns_valid_index(self):
        agent = SelectionAgent(state_dim=45, num_archetypes=10)
        state = torch.randn(1, 45)
        idx = agent.select_archetype(state)

        assert isinstance(idx, int)
        assert 0 <= idx < 10

    def test_returns_valid_index_custom_k(self):
        agent = SelectionAgent(state_dim=20, num_archetypes=5)
        state = torch.randn(1, 20)
        idx = agent.select_archetype(state)

        assert 0 <= idx < 5

    def test_works_with_1d_input(self):
        """支持 (state_dim,) 形状的 1D 输入"""
        agent = SelectionAgent(state_dim=45)
        state = torch.randn(45)  # 1D
        idx = agent.select_archetype(state)

        assert isinstance(idx, int)
        assert 0 <= idx < 10

    def test_works_with_2d_input(self):
        """支持 (1, state_dim) 形状的 2D 输入"""
        agent = SelectionAgent(state_dim=45)
        state = torch.randn(1, 45)  # 2D
        idx = agent.select_archetype(state)

        assert isinstance(idx, int)
        assert 0 <= idx < 10

    def test_multiple_calls_produce_valid_indices(self):
        """多次调用都应返回有效索引"""
        agent = SelectionAgent(state_dim=45, num_archetypes=10)
        for _ in range(20):
            state = torch.randn(45)
            idx = agent.select_archetype(state)
            assert 0 <= idx < 10


class TestSelectionAgentSaveLoad:
    """模型保存/加载一致性"""

    def test_save_load_same_outputs(self):
        """保存 state_dict 后加载到新模型，输出应一致"""
        agent = SelectionAgent(state_dim=45, num_archetypes=10)
        state = torch.randn(4, 45)

        # 原始输出
        agent.eval()
        with torch.no_grad():
            probs_orig, value_orig = agent(state)

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(agent.state_dict(), f.name)
            tmp_path = f.name

        try:
            # 加载到新模型
            agent_loaded = SelectionAgent(state_dim=45, num_archetypes=10)
            agent_loaded.load_state_dict(torch.load(tmp_path, weights_only=True))
            agent_loaded.eval()

            with torch.no_grad():
                probs_loaded, value_loaded = agent_loaded(state)

            assert torch.allclose(probs_orig, probs_loaded, atol=1e-6)
            assert torch.allclose(value_orig, value_loaded, atol=1e-6)
        finally:
            os.unlink(tmp_path)

    def test_save_load_state_dict_keys_match(self):
        """保存和加载的 state_dict 键应完全一致"""
        agent = SelectionAgent(state_dim=45)
        sd = agent.state_dict()

        agent2 = SelectionAgent(state_dim=45)
        sd2 = agent2.state_dict()

        assert set(sd.keys()) == set(sd2.keys())


class TestSelectionAgentGradientFlow:
    """梯度流验证：策略头和价值头都应有梯度"""

    def test_gradient_flows_through_policy_head(self):
        agent = SelectionAgent(state_dim=45)
        state = torch.randn(4, 45)
        action_probs, _ = agent(state)

        loss = action_probs.sum()
        loss.backward()

        assert agent.policy_head.weight.grad is not None
        assert agent.policy_head.bias.grad is not None
        # 共享层也应有梯度
        assert agent.shared[0].weight.grad is not None

    def test_gradient_flows_through_value_head(self):
        agent = SelectionAgent(state_dim=45)
        state = torch.randn(4, 45)
        _, value = agent(state)

        loss = value.sum()
        loss.backward()

        assert agent.value_head.weight.grad is not None
        assert agent.value_head.bias.grad is not None
        # 共享层也应有梯度
        assert agent.shared[0].weight.grad is not None

    def test_gradient_flows_through_both_heads(self):
        """同时对两个头反向传播"""
        agent = SelectionAgent(state_dim=45)
        state = torch.randn(4, 45)
        action_probs, value = agent(state)

        loss = action_probs.sum() + value.sum()
        loss.backward()

        # 策略头
        assert agent.policy_head.weight.grad is not None
        # 价值头
        assert agent.value_head.weight.grad is not None
        # 共享层（应收到来自两个头的梯度）
        assert agent.shared[0].weight.grad is not None
        assert agent.shared[2].weight.grad is not None


# ============================================================================
# Property-Based Tests (hypothesis)
# ============================================================================

from hypothesis import given, settings, strategies as st
from src.phase1.vq_decoder import VQDecoder
import copy


class TestProperty16SelectionAgentOutputRange:
    """Property 16: Selection Agent 输出范围
    索引 ∈ {0,...,K-1}，概率非负且和为 1

    # Feature: archetype-trader, Property 16: Selection Agent 输出范围
    **Validates: Requirements 5.2**
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        num_archetypes=st.sampled_from([5, 10, 20]),
    )
    @settings(max_examples=100)
    def test_prop_action_probs_valid_distribution(self, batch_size, num_archetypes):
        """action_probs 非负且和为 1，对随机状态和不同 K 值成立"""
        state_dim = 45
        agent = SelectionAgent(state_dim=state_dim, num_archetypes=num_archetypes)
        agent.eval()

        state = torch.randn(batch_size, state_dim)
        with torch.no_grad():
            action_probs, value = agent(state)

        # action_probs shape 正确
        assert action_probs.shape == (batch_size, num_archetypes)
        # 非负
        assert (action_probs >= 0).all(), "action_probs contains negative values"
        # 和为 1
        sums = action_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5), (
            f"action_probs sums deviate from 1: {sums}"
        )

    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        num_archetypes=st.sampled_from([5, 10, 20]),
    )
    @settings(max_examples=100)
    def test_prop_select_archetype_in_range(self, batch_size, num_archetypes):
        """select_archetype() 返回索引 ∈ {0,...,K-1}"""
        state_dim = 45
        agent = SelectionAgent(state_dim=state_dim, num_archetypes=num_archetypes)
        agent.eval()

        state = torch.randn(state_dim)
        idx = agent.select_archetype(state)

        assert isinstance(idx, int), f"Expected int, got {type(idx)}"
        assert 0 <= idx < num_archetypes, (
            f"Index {idx} out of range [0, {num_archetypes})"
        )


class TestProperty17FrozenDecoderInvariance:
    """Property 17: 冻结 Decoder 参数不变性
    Phase II/III 中参数与 Phase I 结束时相同

    # Feature: archetype-trader, Property 17: 冻结 Decoder 参数不变性
    **Validates: Requirements 5.3**
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=16),
        num_forward_passes=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_prop_frozen_decoder_params_unchanged(self, batch_size, num_forward_passes):
        """冻结 Decoder 后，多次前向传播不改变参数"""
        state_dim = 45
        code_dim = 16
        hidden_dim = 128
        action_dim = 3
        horizon = 10

        decoder = VQDecoder(
            state_dim=state_dim,
            code_dim=code_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        )

        # 冻结参数（模拟 Phase II/III 中的冻结操作）
        for param in decoder.parameters():
            param.requires_grad = False
        decoder.eval()

        # 保存 Phase I 结束时的参数快照
        original_state = copy.deepcopy(decoder.state_dict())

        # 执行多次前向传播（模拟 Phase II/III 推理）
        for _ in range(num_forward_passes):
            states = torch.randn(batch_size, horizon, state_dim)
            z_q = torch.randn(batch_size, code_dim)
            _ = decoder(states, z_q)

        # 验证参数未改变
        current_state = decoder.state_dict()
        for key in original_state:
            assert torch.equal(original_state[key], current_state[key]), (
                f"Parameter '{key}' changed after forward passes"
            )
