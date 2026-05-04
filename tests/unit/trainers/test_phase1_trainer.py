"""``Phase1Trainer`` 单元测试 (聚焦 paper_strict 行为与 manifest 模式)."""
from __future__ import annotations

import json
from types import SimpleNamespace

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
from src.data.dataset import Phase1DemoDataset
from src.data.demo_store import Phase1DemoStore
from src.data.horizon_builder import HorizonRecord
from src.evaluation.phase1_evaluator import EpochMetrics
from src.trainers.phase1_checkpoint import Phase1CheckpointManager
from src.trainers.phase1_selection_policy import Phase1SelectionPolicy, SelectionHistory
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


def test_phase_a_end_triggers_full_baseline_validation():
    assert Phase1Trainer._is_phase_a_baseline_validation(
        epoch=14,
        is_phase_a=True,
        effective_pretrain_epochs=15,
    )
    assert not Phase1Trainer._is_phase_a_baseline_validation(
        epoch=13,
        is_phase_a=True,
        effective_pretrain_epochs=15,
    )
    assert not Phase1Trainer._is_phase_a_baseline_validation(
        epoch=15,
        is_phase_a=False,
        effective_pretrain_epochs=15,
    )
    assert Phase1Trainer._is_full_validation_epoch(
        epoch=14,
        phase_a_baseline_validation=True,
        full_validation_every_epochs=999,
    )


def test_train_loop_runs_phase_a_baseline_before_phase_b_validation(tmp_path):
    torch = pytest.importorskip("torch")

    class TinyQuantizer:
        num_codes = 2

        def __init__(self):
            self.codebook = torch.ones(2, 2)

        def update_codebook(self, z_e, code_id):
            return None

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.quantizer = TinyQuantizer()

        def _outputs(self, states, actions, *, with_codes: bool):
            batch_size, horizon = actions.shape
            logits = self.weight.expand(batch_size, horizon, 3)
            z_e = (self.weight + 1.0).expand(batch_size, 2)
            code_id = (
                torch.arange(batch_size, device=states.device) % 2
                if with_codes
                else None
            )
            return SimpleNamespace(
                action_logits=logits,
                z_e=z_e,
                z_q_no_grad=z_e,
                code_id=code_id,
            )

        def forward_pretrain(self, states, actions, rewards):
            return self._outputs(states, actions, with_codes=False)

        def forward(self, states, actions, rewards):
            return self._outputs(states, actions, with_codes=True)

    class TinyLoss:
        def _loss(self, action_logits):
            total = action_logits.mean()
            zero = total * 0.0
            return SimpleNamespace(
                total=total,
                reconstruction=total,
                commitment=zero,
                usage=None,
                alignment=None,
            )

        def forward_pretrain(self, *, action_logits, target_actions):
            return self._loss(action_logits)

        def __call__(self, **kwargs):
            return self._loss(kwargs["action_logits"])

    class RecordingEvaluator:
        def __init__(self):
            self.calls = []

        def evaluate_epoch(self, *, epoch, model, val_data, val_records, full_validation):
            self.calls.append((epoch, full_validation))
            stability_measured = len(self.calls) > 1
            metrics = {
                "code_usage_ratio": 1.0,
                "val_max_drawdown": 0.01,
                "val_sharpe_ratio": 1.0,
                "inter_code_action_diversity": 1.0,
                "decoder_sensitivity_to_code": 1.0,
                "epoch_code_stability_measured": stability_measured,
                "epoch_code_stability": 0.9,
                "val_dp_teacher_profitable_ratio": 1.0,
                "switch_point_recall": 1.0,
                "switch_direction_accuracy": 1.0,
                "val_weighted_reconstruction_accuracy": 1.0,
                "val_return_capture_ratio": 1.0,
            }
            return EpochMetrics(
                epoch=epoch,
                metrics=metrics,
                diagnostics={"call_count": len(self.calls)},
            )

    def record(idx: int) -> HorizonRecord:
        return HorizonRecord(
            sample_id=f"r{idx}",
            start_index=idx,
            end_index=idx + 2,
            pair="TEST",
            split="train",
            strata_label="flat|low|mixed",
            states=[[0.1, 0.2], [0.2, 0.3]],
            prices=[100.0, 100.1, 100.2],
            execution_books=[],
            actions=[0, 2],
            rewards=[0.1, -0.1],
        )

    records = [record(i) for i in range(4)]
    config = _config(
        artifact_root=str(tmp_path),
        model=ModelConfig(
            codebook=CodebookConfig(
                init_method="random_normal",
                health=CodebookHealthConfig(dead_code_restart=False),
            )
        ),
        training=TrainingConfig(
            epochs=5,
            pretrain_epochs=2,
            batch_size=2,
            device="cpu",
            mixed_precision=False,
            save_every=10,
        ),
    )
    trainer = Phase1Trainer(config)
    model = TinyModel()
    evaluator = RecordingEvaluator()
    history = trainer._train_loop(
        model=model,
        loss_fn=TinyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        train_dataset=Phase1DemoDataset(records),
        evaluator=evaluator,
        val_dataset=Phase1DemoDataset(records),
        val_records=records,
        checkpoint=Phase1CheckpointManager(tmp_path / "phase1"),
        policy=Phase1SelectionPolicy(config.selection_policy),
        history=SelectionHistory(),
    )

    assert evaluator.calls == [(1, True), (4, True)]
    assert history.best_epoch == 4


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
