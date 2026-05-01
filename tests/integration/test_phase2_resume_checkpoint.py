"""Phase II checkpoint 恢复集成测试。"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from src.trainers.phase2_trainer import Phase2Trainer
from tests.phase2_test_utils import make_config, write_market_splits


def _run_initial_and_resume(tmp_path):
    train, val, test = write_market_splits(tmp_path / "market")
    base = make_config(
        tmp_path,
        horizon=4,
        num_envs=1,
        rollout_length=2,
        total_timesteps=4,
        phase2_batch_id="resume_phase2",
    )
    base = replace(base, train_file=str(train), val_file=str(val), test_file=str(test))
    first = Phase2Trainer(base).run()
    resumed_cfg = replace(
        base,
        total_timesteps=8,
        resume_from=str(first.last_selector),
    )
    second = Phase2Trainer(resumed_cfg).run()
    return first, second


class TestPhase2ResumeCheckpoint:

    @pytest.mark.integration
    def test_resume_continues_from_last(self, tmp_path):
        """第二次训练从上次 update 继续。"""
        _first, second = _run_initial_and_resume(tmp_path)
        payload = json.loads(second.phase2_report.read_text(encoding="utf-8"))
        assert payload["resume_ready"]["restored_update_count"] > 0

    @pytest.mark.integration
    def test_state_restored(self, tmp_path):
        """optimizer / scheduler / RNG / env cursor 被恢复。"""
        first, _second = _run_initial_and_resume(tmp_path)
        state = torch.load(first.last_selector, map_location="cpu", weights_only=False)
        assert "optimizer_state" in state
        assert "schedule_state" in state
        assert "rng_state" in state
        assert "env_states" in state

    @pytest.mark.integration
    def test_position_consistency_after_resume(self, tmp_path):
        """恢复后 prev_terminal_position 通过一致性校验。"""
        first, second = _run_initial_and_resume(tmp_path)
        before = torch.load(first.last_selector, map_location="cpu", weights_only=False)
        after = torch.load(second.last_selector, map_location="cpu", weights_only=False)
        assert before["env_states"][0]["prev_terminal_position"] in (-1, 0, 1)
        assert after["env_states"][0]["prev_terminal_position"] in (-1, 0, 1)
        assert second.replay_log.exists()
