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
from src.data.demo_store import Phase1DemoStore
from src.data.horizon_builder import HorizonRecord
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
    assert new.model.codebook.health.usage_profit_alignment_weight == 0.0
    assert new.model.codebook.health.dead_code_restart is False
    assert new.model.encoder_input.reward_normalization == "train_reward_standard"
    assert new.training.pretrain_epochs == 0


def test_effective_pretrain_epochs_clamps_to_leave_phase_b():
    trainer = Phase1Trainer(
        _config(training=TrainingConfig(epochs=2, pretrain_epochs=10, device="cpu"))
    )
    assert trainer._effective_pretrain_epochs() == 1


def test_build_training_components_wires_alignment_weight(tmp_path):
    pytest.importorskip("torch")
    health = CodebookHealthConfig(
        usage_regularization_weight=0.0,
        usage_profit_alignment_weight=0.07,
        usage_profit_alignment_target_corr=0.4,
        dead_code_restart=False,
    )
    model_cfg = ModelConfig(
        hidden_dim=16,
        code_dim=4,
        num_codes=4,
        codebook=CodebookConfig(init_method="random_normal", health=health),
    )
    trainer = Phase1Trainer(
        _config(
            artifact_root=str(tmp_path),
            model=model_cfg,
            training=TrainingConfig(device="cpu"),
        )
    )
    _, _, loss_fn, _ = trainer._build_training_components(
        feature_dim=2,
        val_horizons=[],
        reward_normalizer=None,
    )
    assert loss_fn.usage_profit_alignment_weight == pytest.approx(0.07)
    assert loss_fn.usage_profit_alignment_target_corr == pytest.approx(0.4)


def test_batch_amp_disabled_when_inputs_exceed_fp16_safe_range():
    torch = pytest.importorskip("torch")
    trainer = Phase1Trainer(_config(training=TrainingConfig(device="cpu")))
    states = torch.tensor([[[1.0e10]]])
    rewards = torch.zeros(1, 1)
    enabled, logged = trainer._batch_amp_enabled(
        base_amp_enabled=True,
        states=states,
        rewards=rewards,
        epoch=0,
        batch_idx=0,
        logged=False,
    )
    assert enabled is False
    assert logged is True


def test_batch_amp_rejects_nonfinite_inputs():
    torch = pytest.importorskip("torch")
    trainer = Phase1Trainer(_config(training=TrainingConfig(device="cpu")))
    states = torch.tensor([[[float("nan")]]])
    rewards = torch.zeros(1, 1)
    with pytest.raises(Phase1FatalError, match="non-finite states"):
        trainer._batch_amp_enabled(
            base_amp_enabled=True,
            states=states,
            rewards=rewards,
            epoch=0,
            batch_idx=0,
            logged=False,
        )


def test_export_horizon_labels_batches_encoder_calls(tmp_path):
    torch = pytest.importorskip("torch")

    class FakeModel:
        def __init__(self):
            self.training = True
            self.batch_sizes = []

        def eval(self):
            self.training = False
            return self

        def train(self):
            self.training = True
            return self

        def encode(self, states, actions, rewards):
            self.batch_sizes.append(int(states.shape[0]))
            ids = torch.arange(states.shape[0], device=states.device) % 3
            return ids, None

    def record(idx: int) -> HorizonRecord:
        return HorizonRecord(
            sample_id=f"r{idx}",
            start_index=idx,
            end_index=idx + 2,
            pair="TEST",
            split="train",
            strata_label="up|low|mixed",
            states=[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            prices=[100.0, 100.1, 100.2, 100.3],
            execution_books=[],
            actions=[1, 2, 2],
            rewards=[0.0, 0.1, 0.2],
        )

    trainer = Phase1Trainer(
        _config(
            artifact_root=str(tmp_path),
            training=TrainingConfig(device="cpu", batch_size=2),
            horizon=3,
        )
    )
    store = Phase1DemoStore(tmp_path / "phase1", "cfg", "schema")
    model = FakeModel()
    paths = trainer._export_horizon_labels(
        model,
        store=store,
        horizons_by_split={"train": [record(i) for i in range(5)]},
        normalizer=None,
    )

    labels = store.load_labels("train")
    assert paths["train"].exists()
    assert model.batch_sizes == [2, 2, 1]
    assert model.training is True
    assert [label.code_label for label in labels] == [0, 1, 0, 1, 0]


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


def test_sampling_leakage_diagnostics_missing_prospective_report_no_throw(tmp_path):
    config = _config(
        artifact_root=str(tmp_path),
        stratification=StratificationConfig(
            mode="hindsight_horizon",
            diagnostic_pair_batch_id="nonexistent_batch",
            hindsight_vs_prospective_max_delta={
                "val_return_capture_ratio": 0.20,
            },
        ),
    )
    trainer = Phase1Trainer(config)
    payload = trainer._build_sampling_leakage_diagnostics(
        {"val_return_capture_ratio": 0.50}
    )
    assert payload["hindsight_bias_warning"] == "missing_prospective_report"
    assert payload["signoff_blocked_reason"] == "missing_prospective_report"


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
