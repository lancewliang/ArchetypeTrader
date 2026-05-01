"""Phase II numerical safety 单元测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.config.phase2_config import NumericalSafetyConfig
from src.rl.ppo_trainer import NumericalSafetyError
from tests.unit.rl.test_ppo_trainer import _trainer


class TestPhase2NumericalSafety:

    def test_non_finite_tensor_fail_fast(self, tmp_path):
        """非 finite tensor 触发 fail-fast。"""
        trainer = _trainer(tmp_path)
        with pytest.raises(NumericalSafetyError):
            trainer._check_numerical_safety(torch.tensor(float("inf")))

    def test_gradient_norm_explosion_fail_fast(self, tmp_path):
        """gradient norm 爆炸触发 fail-fast。"""
        trainer = _trainer(
            tmp_path,
            numerical_safety=NumericalSafetyConfig(max_gradient_norm=0.001),
        )
        for param in trainer.actor_critic.selector.parameters():
            param.grad = torch.ones_like(param)
            break
        with pytest.raises(NumericalSafetyError):
            trainer._check_gradient_safety()

    def test_debug_snapshot_exported(self, tmp_path):
        """debug snapshot 路径被写出。"""
        trainer = _trainer(tmp_path)
        with pytest.raises(NumericalSafetyError):
            trainer._check_numerical_safety(torch.tensor(float("nan")))
        assert list(trainer.config.artifacts_dir().glob("debug_snapshots/*.pt"))
