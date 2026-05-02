"""``Phase1Trainer`` 单元测试 (聚焦 paper_strict 行为与 manifest 模式)."""
from __future__ import annotations

import json

import pytest

from src.config.phase1_config import (
    CodebookConfig,
    CodebookHealthConfig,
    DPConfig,
    DataAugmentationConfig,
    EncoderInputConfig,
    ModelConfig,
    Phase1Config,
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
        data_process_manifest="/tmp/manifest.json",
        artifact_root="/tmp/artifacts",
    )
    base.update(overrides)
    return Phase1Config(**base)


def test_apply_paper_strict_disables_engineering_options():
    cfg = _config(training=TrainingConfig(paper_strict_reproduction=True))
    new = apply_paper_strict_overrides(cfg)
    assert new.model.codebook.update_method == "gradient"
    assert new.model.codebook.health.usage_regularization_weight == 0.0
    assert new.model.codebook.health.dead_code_restart is False
    assert new.model.encoder_input.reward_normalization == "train_reward_standard"


def test_sampling_leakage_diagnostics_compares_prospective_report(tmp_path):
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


def _minimal_manifest(tmp_path, *, pair: str = "TEST", create_artifacts: bool = True):
    manifest_dir = tmp_path / "processed"
    manifest_dir.mkdir(parents=True)
    if create_artifacts:
        for name in (
            "sampled_horizons_train.feather",
            "dp_teacher_train.feather",
            "sampled_horizons_val.feather",
            "dp_teacher_val.feather",
            "sampled_horizons_test.feather",
            "dp_teacher_test.feather",
        ):
            (manifest_dir / name).touch()
    payload = {
        "version": 1,
        "phase": "phase1_data_process",
        "pair": pair,
        "data_batch_id": "processed",
        "artifact_dir": str(manifest_dir),
        "created_at": "2026-05-02T00:00:00Z",
        "input_files": {},
        "input_schema_path": "input_schema.json",
        "schema_hash": "schema",
        "data_process_hash": "data",
        "dp_teacher_hash": "teacher",
        "feature_source": {},
        "splits": {
            split: {
                "window_index_path": f"window_index_{split}.feather",
                "sampled_horizons_path": f"sampled_horizons_{split}.feather",
                "dp_teacher_path": f"dp_teacher_{split}.feather",
                "num_horizons": 0,
            }
            for split in ("train", "val", "test")
        },
    }
    path = manifest_dir / "data_process_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase1_trainer_manifest_pair_mismatch_fails(tmp_path):
    manifest = _minimal_manifest(tmp_path, pair="OTHER")
    trainer = Phase1Trainer(
        _config(
            artifact_root=str(tmp_path / "artifacts"),
            data_process_manifest=str(manifest),
        )
    )

    with pytest.raises(Phase1FatalError, match="pair mismatch"):
        trainer.run()


def test_phase1_trainer_manifest_missing_file_fails(tmp_path):
    manifest = _minimal_manifest(tmp_path, pair="TEST", create_artifacts=False)
    trainer = Phase1Trainer(
        _config(
            artifact_root=str(tmp_path / "artifacts"),
            data_process_manifest=str(manifest),
        )
    )

    with pytest.raises(FileNotFoundError):
        trainer.run()


def test_train_phase1_cli_sets_dp_max_position_10():
    from scripts.train_phase1 import build_config, build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--pair",
            "AL",
            "--train-batch-id",
            "batch",
            "--data-process-manifest",
            "/tmp/manifest.json",
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
            "--data-process-manifest",
            "/tmp/manifest.json",
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
            "--data-process-manifest",
            "/tmp/manifest.json",
            "--local-smoke-relaxed-guardrails",
        ]
    )
    config = build_config(args)
    assert config.selection_policy.min_code_usage_ratio == 0.0
    assert config.selection_policy.behavior.min_inter_code_action_diversity == 0.0
    assert config.training.full_validation_every_epochs == 1
    assert config.local_smoke_relaxed_guardrails is True


def test_training_config_hash_differs_from_config_hash():
    config = _config()
    assert config.training_config_hash() != config.config_hash()


def test_training_config_hash_stable():
    config = _config()
    assert config.training_config_hash() == config.training_config_hash()
