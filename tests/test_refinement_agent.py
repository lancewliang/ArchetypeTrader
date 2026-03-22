"""Refinement Agent 单元测试"""

import torch
import pytest

from src.phase3.refinement_agent import RefinementAgent


class TestRefinementAgent:
    """RefinementAgent 基本功能测试。"""

    def setup_method(self):
        """初始化测试用的 agent 和默认维度。"""
        self.market_dim = 45
        self.context_dim = 19  # code_dim(16) + a_base(1) + R_arche(1) + τ_remain(1)
        self.agent = RefinementAgent(
            market_dim=self.market_dim,
            context_dim=self.context_dim,
        )

    def test_output_shapes(self):
        """测试输出 shape 正确性。"""
        batch = 8
        s_ref1 = torch.randn(batch, self.market_dim)
        s_ref2 = torch.randn(batch, self.context_dim)

        action_probs, value = self.agent(s_ref1, s_ref2)

        assert action_probs.shape == (batch, 3)
        assert value.shape == (batch, 1)

    def test_action_probs_valid_distribution(self):
        """测试策略输出为有效概率分布（非负且和为 1）。"""
        batch = 4
        s_ref1 = torch.randn(batch, self.market_dim)
        s_ref2 = torch.randn(batch, self.context_dim)

        action_probs, _ = self.agent(s_ref1, s_ref2)

        # 非负
        assert (action_probs >= 0).all()
        # 和为 1
        sums = action_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch), atol=1e-6)

    def test_single_sample(self):
        """测试单样本输入。"""
        s_ref1 = torch.randn(1, self.market_dim)
        s_ref2 = torch.randn(1, self.context_dim)

        action_probs, value = self.agent(s_ref1, s_ref2)

        assert action_probs.shape == (1, 3)
        assert value.shape == (1, 1)

    def test_gradient_flow(self):
        """测试梯度能正常回传。"""
        s_ref1 = torch.randn(4, self.market_dim)
        s_ref2 = torch.randn(4, self.context_dim)

        action_probs, value = self.agent(s_ref1, s_ref2)
        loss = -action_probs.log().mean() + value.mean()
        loss.backward()

        # 检查所有参数都有梯度
        for name, param in self.agent.named_parameters():
            assert param.grad is not None, f"参数 {name} 没有梯度"

    def test_adaln_conditioning_effect(self):
        """测试 AdaLN 条件化确实影响输出 — 不同 s_ref2 应产生不同输出。"""
        s_ref1 = torch.randn(1, self.market_dim)
        s_ref2_a = torch.randn(1, self.context_dim)
        s_ref2_b = torch.randn(1, self.context_dim) * 10  # 明显不同的条件

        with torch.no_grad():
            probs_a, val_a = self.agent(s_ref1, s_ref2_a)
            probs_b, val_b = self.agent(s_ref1, s_ref2_b)

        # 不同条件应产生不同输出
        assert not torch.allclose(probs_a, probs_b, atol=1e-6)

    def test_adaln_gamma_beta_affect_output(self):
        """验证 AdaLN 内部的 gamma/beta 投影确实被 s_ref2 驱动。

        固定 s_ref1，用零向量 vs 非零向量作为 s_ref2，
        检查 AdaLN 层输出的 gamma 和 beta 不同。
        """
        s_ref1 = torch.randn(1, self.market_dim)
        s_ref2_zero = torch.zeros(1, self.context_dim)
        s_ref2_nonzero = torch.ones(1, self.context_dim) * 5.0

        adaln = self.agent.adaln
        with torch.no_grad():
            encoded = self.agent.market_encoder(s_ref1)
            gamma_zero = adaln.gamma_proj(s_ref2_zero)
            beta_zero = adaln.beta_proj(s_ref2_zero)
            gamma_nonzero = adaln.gamma_proj(s_ref2_nonzero)
            beta_nonzero = adaln.beta_proj(s_ref2_nonzero)

        assert not torch.allclose(gamma_zero, gamma_nonzero, atol=1e-6), \
            "gamma should differ for different conditions"
        assert not torch.allclose(beta_zero, beta_nonzero, atol=1e-6), \
            "beta should differ for different conditions"

    def test_adaln_zero_condition_deterministic(self):
        """相同的零条件向量应产生完全相同的输出（确定性）。"""
        s_ref1 = torch.randn(1, self.market_dim)
        s_ref2 = torch.zeros(1, self.context_dim)

        with torch.no_grad():
            probs_a, val_a = self.agent(s_ref1, s_ref2)
            probs_b, val_b = self.agent(s_ref1, s_ref2)

        assert torch.allclose(probs_a, probs_b, atol=1e-7)
        assert torch.allclose(val_a, val_b, atol=1e-7)

    def test_adaln_conditioning_propagates_to_policy_and_value(self):
        """不同 s_ref2 应同时影响策略头和价值头的输出。"""
        s_ref1 = torch.randn(2, self.market_dim)
        s_ref2_a = torch.randn(2, self.context_dim)
        s_ref2_b = torch.randn(2, self.context_dim) * -3.0

        with torch.no_grad():
            probs_a, val_a = self.agent(s_ref1, s_ref2_a)
            probs_b, val_b = self.agent(s_ref1, s_ref2_b)

        assert not torch.allclose(probs_a, probs_b, atol=1e-5), \
            "Policy head should differ for different conditions"
        assert not torch.allclose(val_a, val_b, atol=1e-5), \
            "Value head should differ for different conditions"

    def test_adaln_batch_independence(self):
        """批次内不同样本的条件化应独立——改变一个样本的 s_ref2 不影响另一个。"""
        s_ref1 = torch.randn(2, self.market_dim)
        s_ref2_orig = torch.randn(2, self.context_dim)

        # 只改变第二个样本的 s_ref2
        s_ref2_modified = s_ref2_orig.clone()
        s_ref2_modified[1] = torch.randn(self.context_dim) * 10.0

        with torch.no_grad():
            probs_orig, val_orig = self.agent(s_ref1, s_ref2_orig)
            probs_mod, val_mod = self.agent(s_ref1, s_ref2_modified)

        # 第一个样本的输出应不变
        assert torch.allclose(probs_orig[0], probs_mod[0], atol=1e-6), \
            "Changing s_ref2 of sample 1 should not affect sample 0"
        assert torch.allclose(val_orig[0], val_mod[0], atol=1e-6)

        # 第二个样本的输出应改变
        assert not torch.allclose(probs_orig[1], probs_mod[1], atol=1e-5)

    def test_custom_dimensions(self):
        """测试非默认维度的 RefinementAgent 正常工作。"""
        agent = RefinementAgent(market_dim=20, context_dim=10)
        s_ref1 = torch.randn(3, 20)
        s_ref2 = torch.randn(3, 10)
        probs, val = agent(s_ref1, s_ref2)
        assert probs.shape == (3, 3)
        assert val.shape == (3, 1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-6)

    def test_export_from_package(self):
        """测试从 phase3 包导入 RefinementAgent。"""
        from src.phase3 import RefinementAgent as RA
        assert RA is RefinementAgent


# ---------------------------------------------------------------------------
# Property-Based Tests (hypothesis)
# ---------------------------------------------------------------------------
from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np


class TestRefinementAgentProperties:
    """RefinementAgent 属性测试。"""

    @given(
        batch=st.integers(min_value=1, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100)
    def test_prop19_refinement_agent_output_range(self, batch, seed):
        """Property 19: Refinement Agent 输出范围 — a_ref ∈ {-1, 0, 1}。

        **Validates: Requirements 6.3**

        # Feature: archetype-trader, Property 19: Refinement Agent 输出范围

        验证:
        - action_probs shape 为 (batch, 3)，非负，和为 1
        - argmax 产生索引 ∈ {0, 1, 2}（映射到 a_ref ∈ {-1, 0, 1}）
        """
        market_dim = 45
        context_dim = 19
        agent = RefinementAgent(market_dim=market_dim, context_dim=context_dim)

        # Generate random inputs using seed for reproducibility
        rng = np.random.RandomState(seed)
        s_ref1 = torch.from_numpy(rng.randn(batch, market_dim).astype(np.float32))
        s_ref2 = torch.from_numpy(rng.randn(batch, context_dim).astype(np.float32))

        with torch.no_grad():
            action_probs, value = agent(s_ref1, s_ref2)

        # Shape checks
        assert action_probs.shape == (batch, 3), (
            f"Expected action_probs shape ({batch}, 3), got {action_probs.shape}"
        )

        # Non-negative
        assert (action_probs >= 0).all(), "action_probs should be non-negative"

        # Sum to 1
        sums = action_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch), atol=1e-5), (
            f"action_probs should sum to 1, got sums={sums}"
        )

        # Argmax produces indices in {0, 1, 2}
        indices = torch.argmax(action_probs, dim=-1)
        for idx in indices:
            assert idx.item() in {0, 1, 2}, (
                f"argmax index should be in {{0, 1, 2}}, got {idx.item()}"
            )

        # Mapping: index 0 → a_ref=-1, index 1 → a_ref=0, index 2 → a_ref=1
        a_ref_map = {0: -1, 1: 0, 2: 1}
        for idx in indices:
            a_ref = a_ref_map[idx.item()]
            assert a_ref in {-1, 0, 1}, (
                f"Mapped a_ref should be in {{-1, 0, 1}}, got {a_ref}"
            )
