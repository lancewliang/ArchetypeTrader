"""``Phase1Trainer`` 单元测试 (聚焦 prospective 对照与 paper_strict 行为)."""
from __future__ import annotations

import pytest

from src.config.phase1_config import (
    CodebookConfig,
    CodebookHealthConfig,
    DPConfig,
    DataAugmentationConfig,
    EncoderInputConfig,
    ModelConfig,
    Phase1Config,
    SamplingHealthConfig,
    SelectionPolicyConfig,
    StratificationConfig,
    TrainingConfig,
    apply_paper_strict_overrides,
)
from src.trainers.phase1_trainer import Phase1FatalError, Phase1Trainer


def _config(**overrides) -> Phase1Config:
    base = dict(
        pair="TEST",
        train_batch_id="batch_001",
        train_file="train.feather",
        val_file="val.feather",
        test_file="test.feather",
        artifact_root="/tmp/artifacts",
    )
    base.update(overrides)
    return Phase1Config(**base)


def test_check_prospective_blocks_when_missing():
    config = _config(stratification=StratificationConfig(mode="hindsight_horizon", diagnostic_pair_batch_id=None))
    trainer = Phase1Trainer(config)
    with pytest.raises(Phase1FatalError):
        trainer._check_prospective_diagnostic()


def test_check_prospective_allows_when_explicit_acknowledgment():
    config = _config(
        stratification=StratificationConfig(mode="hindsight_horizon", diagnostic_pair_batch_id=None),
        allow_missing_prospective_diagnostic=True,
        risk_acknowledged_by="alice",
        expected_sign_off_followup_batch_id="batch_followup",
    )
    trainer = Phase1Trainer(config)
    trainer._check_prospective_diagnostic()  # 不应抛错


def test_check_prospective_skips_for_diagnostic_batch():
    config = _config(stratification=StratificationConfig(mode="prospective_past"))
    trainer = Phase1Trainer(config)
    trainer._check_prospective_diagnostic()


def test_apply_paper_strict_disables_engineering_options():
    cfg = _config(training=TrainingConfig(paper_strict_reproduction=True))
    new = apply_paper_strict_overrides(cfg)
    assert new.model.codebook.update_method == "gradient"
    assert new.model.codebook.health.usage_regularization_weight == 0.0
    assert new.model.codebook.health.dead_code_restart is False
    assert new.model.encoder_input.reward_normalization == "train_reward_standard"
