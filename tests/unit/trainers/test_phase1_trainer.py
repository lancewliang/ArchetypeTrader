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


def test_sampling_leakage_diagnostics_compares_prospective_report(tmp_path):
    import json

    diag_dir = tmp_path / "TEST" / "prospective" / "phase1"
    diag_dir.mkdir(parents=True)
    (diag_dir / "phase1_report.json").write_text(
        json.dumps(
            {
                "val_return_capture_ratio": 0.10,
                "val_sharpe_ratio": 0.20,
                "val_max_drawdown": 0.05,
                "code_usage_ratio": 0.80,
            }
        ),
        encoding="utf-8",
    )
    config = _config(
        artifact_root=str(tmp_path),
        stratification=StratificationConfig(
            mode="hindsight_horizon",
            diagnostic_pair_batch_id="prospective",
            hindsight_vs_prospective_max_delta={
                "val_return_capture_ratio": 0.20,
                "val_sharpe_ratio": 0.50,
                "val_max_drawdown": 0.10,
                "code_usage_ratio": 0.10,
            },
        ),
    )
    trainer = Phase1Trainer(config)
    payload = trainer._build_sampling_leakage_diagnostics(
        {
            "val_return_capture_ratio": 0.50,
            "val_sharpe_ratio": 0.20,
            "val_max_drawdown": 0.05,
            "code_usage_ratio": 0.80,
        }
    )
    assert payload["hindsight_bias_warning"] == "exceeded"
    assert payload["hindsight_vs_prospective_metric_delta"]["val_return_capture_ratio"]["exceeded"]


def test_num_samples_for_validation_is_capped_and_scales_down_for_smoke():
    trainer = Phase1Trainer(_config(num_demos=12))
    assert trainer._num_samples_for_split("val", 720) == 1

    trainer = Phase1Trainer(_config(num_demos=30_000))
    assert trainer._num_samples_for_split("test", 10_000) == 64


def test_phase1_trainer_uses_factor_list_schema(tmp_path):
    import polars as pl

    factor_file = tmp_path / "short.txt"
    factor_file.write_text("factor_a\n", encoding="utf-8")
    frame = pl.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "close": [100.0, 101.0, 102.0],
            "ask1_price": [101.0, 102.0, 103.0],
            "ask1_size": [10.0, 10.0, 10.0],
            "bid1_price": [99.0, 100.0, 101.0],
            "bid1_size": [10.0, 10.0, 10.0],
            "ask2_price": [102.0, 103.0, 104.0],
            "ask2_size": [10.0, 10.0, 10.0],
            "bid2_price": [98.0, 99.0, 100.0],
            "bid2_size": [10.0, 10.0, 10.0],
            "ask3_price": [103.0, 104.0, 105.0],
            "ask3_size": [10.0, 10.0, 10.0],
            "bid3_price": [97.0, 98.0, 99.0],
            "bid3_size": [10.0, 10.0, 10.0],
            "ask4_price": [104.0, 105.0, 106.0],
            "ask4_size": [10.0, 10.0, 10.0],
            "bid4_price": [96.0, 97.0, 98.0],
            "bid4_size": [10.0, 10.0, 10.0],
            "ask5_price": [105.0, 106.0, 107.0],
            "ask5_size": [10.0, 10.0, 10.0],
            "bid5_price": [95.0, 96.0, 97.0],
            "bid5_size": [10.0, 10.0, 10.0],
            "total_trade_volume": [100.0, 101.0, 102.0],
            "turnover": [1000.0, 1010.0, 1020.0],
            "open_interest": [50.0, 51.0, 52.0],
            "factor_a": [0.1, 0.2, 0.3],
            "unused_numeric_feature": [9.0, 9.0, 9.0],
        }
    )
    trainer = Phase1Trainer(
        _config(factor_list_file=str(factor_file), factor_profile="short")
    )
    schema = trainer._build_schema_validator().validate(frame)
    assert schema.feature_columns[-1] == "factor_a"
    assert "unused_numeric_feature" not in schema.feature_columns
    assert schema.feature_source["mode"] == "fixed_plus_factor_list"


def test_phase1_trainer_rejects_missing_factor_column(tmp_path):
    import polars as pl

    factor_file = tmp_path / "short.txt"
    factor_file.write_text("factor_a\n", encoding="utf-8")
    frame = pl.DataFrame({"timestamp": [0, 1], "close": [100.0, 101.0]})
    trainer = Phase1Trainer(_config(factor_list_file=str(factor_file)))
    with pytest.raises(ValueError, match="ask1_price"):
        trainer._build_schema_validator().validate(frame)


def test_train_phase1_cli_sets_dp_max_position_10():
    from scripts.train_phase1 import build_config, build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--pair",
            "AL",
            "--train-batch-id",
            "batch",
            "--train-file",
            "data/AL/df_train.feather",
            "--val-file",
            "data/AL/df_val.feather",
            "--test-file",
            "data/AL/df_test.feather",
            "--factor-profile",
            "short",
            "--factor-list-file",
            "src/factors/AL/short.txt",
            "--max-position",
            "10",
        ]
    )
    config = build_config(args)
    assert config.factor_profile == "short"
    assert config.factor_list_file == "src/factors/AL/short.txt"
    assert config.dp.max_position == 10


def test_train_phase1_cli_sets_prospective_lookback_minutes():
    from scripts.train_phase1 import build_config, build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--pair",
            "TEST",
            "--train-batch-id",
            "batch",
            "--train-file",
            "train.feather",
            "--val-file",
            "val.feather",
            "--test-file",
            "test.feather",
            "--stratification-mode",
            "prospective_past",
            "--prospective-lookback-minutes",
            "60",
        ]
    )
    config = build_config(args)
    assert config.stratification.mode == "prospective_past"
    assert config.stratification.prospective_lookback_minutes == 60


def test_train_phase1_cli_local_smoke_relaxes_guardrails():
    from scripts.train_phase1 import build_config, build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--pair",
            "TEST",
            "--train-batch-id",
            "smoke",
            "--train-file",
            "train.feather",
            "--val-file",
            "val.feather",
            "--test-file",
            "test.feather",
            "--local-smoke-relaxed-guardrails",
        ]
    )
    config = build_config(args)
    assert config.sampling_health.warn_only is True
    assert config.sampling_health.min_gap_between_samples == 1
    assert config.selection_policy.min_code_usage_ratio == 0.0
    assert config.selection_policy.behavior.min_inter_code_action_diversity == 0.0
    assert config.training.full_validation_every_epochs == 1
    assert config.local_smoke_relaxed_guardrails is True
