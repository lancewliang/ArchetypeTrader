"""Schedule 单元测试。

测试用例:
- lr schedule 生效。
- entropy coef anneal 生效。
- kl_demo coef anneal 生效。
"""
import pytest
import torch

from src.config.phase2_config import Phase2Config, PPOConfig
from src.rl.scheduling import ScheduleManager


@pytest.fixture
def schedule():
    config = Phase2Config(
        ppo=PPOConfig(lr=1e-3, entropy_coef=0.01, kl_demo_coef=0.1)
    )
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return ScheduleManager(config, optimizer, total_updates=100)


class TestScheduleManager:

    def test_lr_schedule(self, schedule):
        """lr schedule 生效: 随 update 递减。"""
        state_0 = schedule.current_state()
        schedule.step(50)
        state_50 = schedule.current_state()
        schedule.step(99)
        state_99 = schedule.current_state()
        assert state_50.lr < state_0.lr
        assert state_99.lr < state_50.lr

    def test_entropy_coef_anneal(self, schedule):
        """entropy coef anneal 生效。"""
        state_0 = schedule.current_state()
        schedule.step(50)
        state_50 = schedule.current_state()
        assert state_50.entropy_coef < state_0.entropy_coef

    def test_kl_demo_coef_anneal(self, schedule):
        """kl_demo coef anneal 生效。"""
        state_0 = schedule.current_state()
        schedule.step(50)
        state_50 = schedule.current_state()
        assert state_50.kl_demo_coef <= state_0.kl_demo_coef

    def test_progress_tracking(self, schedule):
        """progress 正确追踪。"""
        schedule.step(0)
        assert schedule.current_state().progress == pytest.approx(0.0)
        schedule.step(50)
        assert schedule.current_state().progress == pytest.approx(0.5)
        schedule.step(100)
        assert schedule.current_state().progress == pytest.approx(1.0)

    def test_get_load_state(self, schedule):
        """get_state / load_state 往返一致。"""
        schedule.step(42)
        state = schedule.get_state()
        assert state["current_update"] == 42

        # 创建新 schedule 并恢复
        config = Phase2Config(ppo=PPOConfig(lr=1e-3))
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        new_schedule = ScheduleManager(config, opt, total_updates=100)
        new_schedule.load_state(state)
        assert new_schedule._current_update == 42

    def test_entropy_min_coef_floor(self):
        """entropy coef 退火不低于下界。"""
        config = Phase2Config(
            ppo=PPOConfig(
                lr=1e-3,
                entropy_coef=0.01,
                entropy_min_coef=0.002,
            )
        )
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        schedule = ScheduleManager(config, optimizer, total_updates=100)
        schedule.step(100)
        assert schedule.current_state().entropy_coef == pytest.approx(0.002)
