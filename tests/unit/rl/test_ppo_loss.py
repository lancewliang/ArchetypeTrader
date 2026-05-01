"""PPO loss 单元测试。

测试用例:
- clip surrogate 正确。
- value loss 正确。
- entropy bonus 正确。
- kl_demo_loss 只在 is_labeled=true 上生效。
- masked KL label 样本 loss=0。
"""
import pytest
import torch

from src.rl.ppo_loss import PPOLoss


@pytest.fixture
def loss_fn():
    return PPOLoss(clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, kl_demo_coef=0.1, num_codes=5)


class TestPPOLoss:

    def test_clip_surrogate(self, loss_fn):
        """clip surrogate 正确: ratio 在 clip 范围内时 loss 有限。"""
        batch = 8
        log_prob = torch.zeros(batch)
        old_log_prob = torch.zeros(batch)
        advantage = torch.randn(batch)
        value = torch.randn(batch)
        return_ = torch.randn(batch)
        entropy = torch.ones(batch)

        out = loss_fn.compute(log_prob, old_log_prob, advantage, value, return_, entropy)
        assert torch.isfinite(out.total)
        assert torch.isfinite(out.policy_loss)

    def test_value_loss(self, loss_fn):
        """value loss 正确: value 与 return 相同时 loss=0。"""
        batch = 8
        value = torch.tensor([1.0] * batch)
        return_ = torch.tensor([1.0] * batch)
        out = loss_fn.compute(
            torch.zeros(batch), torch.zeros(batch),
            torch.zeros(batch), value, return_, torch.ones(batch),
        )
        assert out.value_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_entropy_bonus(self, loss_fn):
        """entropy bonus 正确: 高 entropy 时 entropy_loss 更负。"""
        batch = 8
        high_entropy = torch.ones(batch) * 2.0
        low_entropy = torch.ones(batch) * 0.1
        out_high = loss_fn.compute(
            torch.zeros(batch), torch.zeros(batch),
            torch.zeros(batch), torch.zeros(batch),
            torch.zeros(batch), high_entropy,
        )
        out_low = loss_fn.compute(
            torch.zeros(batch), torch.zeros(batch),
            torch.zeros(batch), torch.zeros(batch),
            torch.zeros(batch), low_entropy,
        )
        # 高 entropy 的 entropy_loss 更负（鼓励探索）
        assert out_high.entropy_loss.item() < out_low.entropy_loss.item()

    def test_kl_demo_only_labeled(self, loss_fn):
        """kl_demo_loss 只在 is_labeled=true 上生效。"""
        batch = 8
        logits = torch.randn(batch, 5)
        kl_label = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        is_labeled = torch.tensor([True, True, False, False, True, True, False, False])

        out = loss_fn.compute(
            torch.zeros(batch), torch.zeros(batch),
            torch.zeros(batch), torch.zeros(batch),
            torch.zeros(batch), torch.ones(batch),
            kl_label=kl_label, is_labeled=is_labeled, logits=logits,
        )
        assert out.kl_demo_loss.item() > 0  # 有 labeled 样本

    def test_masked_kl_label_zero(self, loss_fn):
        """全部 is_labeled=false 时 kl_demo_loss=0。"""
        batch = 4
        logits = torch.randn(batch, 5)
        kl_label = torch.tensor([0, 1, 2, 3])
        is_labeled = torch.tensor([False, False, False, False])

        out = loss_fn.compute(
            torch.zeros(batch), torch.zeros(batch),
            torch.zeros(batch), torch.zeros(batch),
            torch.zeros(batch), torch.ones(batch),
            kl_label=kl_label, is_labeled=is_labeled, logits=logits,
        )
        assert out.kl_demo_loss.item() == pytest.approx(0.0)

    def test_approx_kl_zero_when_same(self, loss_fn):
        """相同 log_prob 时 approx_kl=0。"""
        batch = 8
        lp = torch.zeros(batch)
        out = loss_fn.compute(lp, lp, torch.zeros(batch), torch.zeros(batch),
                              torch.zeros(batch), torch.ones(batch))
        assert out.approx_kl == pytest.approx(0.0, abs=1e-5)

    def test_clip_fraction(self, loss_fn):
        """大 ratio 时 clip_fraction > 0。"""
        batch = 8
        old_lp = torch.zeros(batch)
        new_lp = torch.ones(batch)  # ratio = e^1 ≈ 2.7
        out = loss_fn.compute(new_lp, old_lp, torch.ones(batch), torch.zeros(batch),
                              torch.zeros(batch), torch.ones(batch))
        assert out.clip_fraction > 0
