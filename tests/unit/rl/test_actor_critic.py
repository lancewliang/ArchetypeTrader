"""ActorCritic 单元测试。

测试用例:
- act() 返回 action/log_prob/value。
- evaluate_actions() 返回 log_prob/entropy/value。
- 多档仓位编码与 state_dim_breakdown 一致。
"""
import pytest
import torch

from src.config.phase2_config import SelectorNetworkConfig
from src.models.archetype_selector import ArchetypeSelector
from src.rl.actor_critic import ActorCritic


@pytest.fixture
def ac():
    config = SelectorNetworkConfig(hidden_dims=[16])
    selector = ArchetypeSelector(state_dim=4, num_codes=5, config=config)
    return ActorCritic(selector)


class TestActorCritic:

    def test_act_returns_correct_fields(self, ac):
        """act() 返回 action/log_prob/value。"""
        obs = torch.randn(3, 4)
        out = ac.act(obs)
        assert out.action.shape == (3,)
        assert out.log_prob.shape == (3,)
        assert out.value.shape == (3,)
        assert out.action.dtype == torch.long

    def test_evaluate_actions_returns_correct_fields(self, ac):
        """evaluate_actions() 返回 log_prob/entropy/value。"""
        obs = torch.randn(3, 4)
        action = torch.tensor([0, 1, 2])
        out = ac.evaluate_actions(obs, action)
        assert out.log_prob.shape == (3,)
        assert out.entropy.shape == (3,)
        assert out.value.shape == (3,)

    def test_get_value(self, ac):
        """get_value 返回 [batch] value。"""
        obs = torch.randn(3, 4)
        value = ac.get_value(obs)
        assert value.shape == (3,)

    def test_dead_code_mask_applied(self):
        """dead code mask 在 act 中生效。"""
        config = SelectorNetworkConfig(hidden_dims=[16])
        selector = ArchetypeSelector(state_dim=4, num_codes=5, config=config)
        mask = torch.tensor([False, False, True, False, False])
        ac = ActorCritic(selector, dead_code_mask=mask)
        obs = torch.randn(100, 4)
        out = ac.act(obs, deterministic=False)
        # code 2 被 mask，不应被选中
        assert (out.action == 2).sum().item() == 0

    def test_evaluate_log_prob_finite(self, ac):
        """evaluate_actions 的 log_prob 应该是有限值。"""
        obs = torch.randn(5, 4)
        action = torch.tensor([0, 1, 2, 3, 4])
        out = ac.evaluate_actions(obs, action)
        assert torch.all(torch.isfinite(out.log_prob))
