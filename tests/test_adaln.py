"""AdaLN 模块单元测试"""

import torch
import pytest
from src.phase3.adaln import AdaptiveLayerNorm


class TestAdaptiveLayerNorm:
    """AdaptiveLayerNorm 基本功能测试"""

    def test_output_shape(self):
        """输出 shape 应与输入 x 的 shape 一致"""
        feature_dim, condition_dim, batch = 64, 19, 4
        adaln = AdaptiveLayerNorm(feature_dim, condition_dim)
        x = torch.randn(batch, feature_dim)
        cond = torch.randn(batch, condition_dim)
        out = adaln(x, cond)
        assert out.shape == (batch, feature_dim)

    def test_different_conditions_produce_different_outputs(self):
        """不同条件向量应产生不同输出"""
        feature_dim, condition_dim = 32, 16
        adaln = AdaptiveLayerNorm(feature_dim, condition_dim)
        x = torch.randn(1, feature_dim)
        c1 = torch.randn(1, condition_dim)
        c2 = torch.randn(1, condition_dim)
        out1 = adaln(x, c1)
        out2 = adaln(x, c2)
        assert not torch.allclose(out1, out2), "Different conditions should yield different outputs"

    def test_gradient_flow(self):
        """梯度应能流过 AdaLN 到 x 和 condition"""
        feature_dim, condition_dim = 16, 8
        adaln = AdaptiveLayerNorm(feature_dim, condition_dim)
        x = torch.randn(2, feature_dim, requires_grad=True)
        cond = torch.randn(2, condition_dim, requires_grad=True)
        out = adaln(x, cond)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow to x"
        assert cond.grad is not None, "Gradient should flow to condition"

    def test_batch_size_one(self):
        """batch_size=1 应正常工作"""
        adaln = AdaptiveLayerNorm(10, 5)
        x = torch.randn(1, 10)
        cond = torch.randn(1, 5)
        out = adaln(x, cond)
        assert out.shape == (1, 10)

    def test_formula_correctness(self):
        """验证 AdaLN(x, c) = γ(c) × LayerNorm(x) + β(c)"""
        feature_dim, condition_dim = 8, 4
        adaln = AdaptiveLayerNorm(feature_dim, condition_dim)
        x = torch.randn(3, feature_dim)
        cond = torch.randn(3, condition_dim)

        # 手动计算
        normalized = adaln.layer_norm(x)
        gamma = adaln.gamma_proj(cond)
        beta = adaln.beta_proj(cond)
        expected = gamma * normalized + beta

        actual = adaln(x, cond)
        assert torch.allclose(actual, expected, atol=1e-6)
