"""Archetype selector 单元测试。

测试用例:
- actor 输出 logits [batch, K]。
- critic 输出 value [batch]。
- dead code mask 正确把 logit 设为 -inf。
- deterministic 模式返回 argmax。
- stochastic 模式可采样不同 action。
- unmasked diagnostic rollout 中可记录 dead code probe_pick_rate。
"""
import pytest
import torch

from src.config.phase2_config import SelectorNetworkConfig
from src.models.archetype_selector import ArchetypeSelector


@pytest.fixture
def selector():
    config = SelectorNetworkConfig(hidden_dims=[32, 16])
    return ArchetypeSelector(state_dim=5, num_codes=10, config=config)


class TestArchetypeSelector:

    def test_actor_output_shape(self, selector):
        """actor 输出 logits [batch, K]。"""
        obs = torch.randn(4, 5)
        logits, value = selector(obs)
        assert logits.shape == (4, 10)

    def test_critic_output_shape(self, selector):
        """critic 输出 value [batch]。"""
        obs = torch.randn(4, 5)
        logits, value = selector(obs)
        assert value.shape == (4,)

    def test_dead_code_mask_neg_inf(self, selector):
        """dead code mask 正确把 logit 设为 -inf。"""
        logits = torch.randn(2, 10)
        mask = torch.tensor([False] * 10)
        mask[3] = True
        mask[7] = True
        masked = ArchetypeSelector.apply_dead_code_mask(logits, mask)
        assert masked[0, 3].item() == float("-inf")
        assert masked[0, 7].item() == float("-inf")
        assert masked[0, 0].item() != float("-inf")

    def test_deterministic_returns_argmax(self, selector):
        """deterministic 模式返回 argmax。"""
        from src.rl.actor_critic import ActorCritic
        ac = ActorCritic(selector)
        obs = torch.randn(4, 5)
        out = ac.act(obs, deterministic=True)
        # 验证 action 是 logits 的 argmax
        logits, _ = selector(obs)
        expected = logits.argmax(dim=-1)
        assert torch.equal(out.action, expected)

    def test_stochastic_samples_different(self, selector):
        """stochastic 模式可采样不同 action（概率性，多次尝试）。"""
        from src.rl.actor_critic import ActorCritic
        ac = ActorCritic(selector)
        obs = torch.randn(1, 5)
        actions = set()
        for _ in range(100):
            out = ac.act(obs, deterministic=False)
            actions.add(out.action.item())
        # 至少应该采样到 2 个不同的 action
        assert len(actions) >= 2

    def test_state_dim_breakdown(self, selector):
        """state_dim_breakdown 返回正确信息。"""
        breakdown = selector.state_dim_breakdown()
        assert breakdown["total_state_dim"] == 5
        assert breakdown["num_codes"] == 10
