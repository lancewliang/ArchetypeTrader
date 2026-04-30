"""Phase I 端到端 smoke test."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
polars = pytest.importorskip("polars")

# 确保 src 在 sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.fixtures.phase1.build_fixtures import FixtureSpec, build_fixtures  # noqa: E402

from src.config.phase1_config import (  # noqa: E402
    CodebookConfig,
    CodebookHealthConfig,
    CostConfig,
    DPConfig,
    DataAugmentationConfig,
    DiagnosticsConfig,
    EncoderInputConfig,
    ModelConfig,
    NoTradeCodeHealthConfig,
    NoTradeControlConfig,
    Phase1Config,
    SamplingHealthConfig,
    SelectionPolicyConfig,
    StratificationConfig,
    TrainingConfig,
)
from src.trainers.phase1_trainer import Phase1Trainer  # noqa: E402


def _make_smoke_config(tmp_path: Path, train_file, val_file, test_file) -> Phase1Config:
    """构造小规模 smoke 配置: h=8, num_demos=12, K=4, epochs=2。"""
    cost = CostConfig(reward_alignment="paper_formula")
    dp = DPConfig(horizon=8, cost_config=cost, max_position=1, gamma=1.0)
    enc = EncoderInputConfig(state_adapter_dim=8, action_embedding_dim=4, reward_embedding_dim=4, fusion_dim=16)
    cb = CodebookConfig(
        init_method="random_normal",
        kmeans_warmup_batches=1,
        update_method="ema",
        health=CodebookHealthConfig(
            usage_regularization_weight=0.0,
            dead_code_restart=False,
            consecutive_collapse_epoch_limit=999,
        ),
    )
    model = ModelConfig(hidden_dim=16, code_dim=4, num_codes=4, encoder_input=enc, codebook=cb)
    training = TrainingConfig(
        batch_size=4, lr=1e-3, epochs=2, seed=7, device="cpu",
        save_every=1, full_validation_every_epochs=1, fast_val_probe_size=8,
    )
    sampling = SamplingHealthConfig(
        max_no_trade_ratio=1.0,
        flat_low_vol_max_ratio=1.0,
        min_gap_between_samples=1,
        max_overlap_ratio=1.0,
        split_boundary_embargo=0,
        next_row_split_boundary_embargo=0,
        warn_only=True,
    )
    selection = SelectionPolicyConfig(
        min_code_usage_ratio=0.0,
    )
    stratification = StratificationConfig(
        mode="hindsight_horizon",
        diagnostic_pair_batch_id="diagnostic_batch",
    )
    return Phase1Config(
        pair="TEST",
        train_batch_id="smoke",
        train_file=str(train_file),
        val_file=str(val_file),
        test_file=str(test_file),
        artifact_root=str(tmp_path / "artifacts"),
        horizon=8,
        num_demos=12,
        sampling_strategy="stratified_uniform",
        stratification=stratification,
        sampling_health=sampling,
        no_trade_control=NoTradeControlConfig(),
        no_trade_code_health=NoTradeCodeHealthConfig(),
        data_augmentation=DataAugmentationConfig(),
        dp=dp,
        model=model,
        training=training,
        selection_policy=selection,
        diagnostics=DiagnosticsConfig(failure_cases_enabled=False, latent_visualization_enabled=False),
    )


@pytest.fixture
def smoke_artifacts(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    train, val, test = build_fixtures(
        fixtures_dir, FixtureSpec(train_rows=400, val_rows=200, test_rows=200)
    )
    diagnostic_dir = tmp_path / "artifacts" / "TEST" / "diagnostic_batch" / "phase1"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    (diagnostic_dir / "phase1_report.json").write_text(
        json.dumps(
            {
                "val_return_capture_ratio": 0.0,
                "val_sharpe_ratio": 0.0,
                "val_max_drawdown": 0.0,
                "code_usage_ratio": 1.0,
                "phase1_composite_score": 0.0,
            }
        ),
        encoding="utf-8",
    )
    config = _make_smoke_config(tmp_path, train, val, test)
    trainer = Phase1Trainer(config)
    return trainer.run()


def test_pipeline_smoke_writes_required_artifacts(smoke_artifacts):
    a = smoke_artifacts
    for path in [
        a.phase1_config_yaml,
        a.input_schema_json,
        a.window_index_train,
        a.demos_train,
        a.horizon_labels_train,
        a.horizon_labels_val,
        a.horizon_labels_test,
        a.best_vq_model,
        a.last_vq_model,
        a.encoder_pt,
        a.decoder_pt,
        a.codebook_pt,
        a.checkpoint_manifest,
        a.phase1_report,
        a.composite_score_sensitivity_json,
        a.sampling_leakage_diagnostics_json,
        a.artifacts_dir / "action_diagnostics.json",
        a.artifacts_dir / "horizon_boundary_diagnostics.json",
        a.artifacts_dir / "code_stability_diagnostics.json",
    ]:
        assert path.exists(), f"missing artifact: {path}"


def test_phase1_report_required_keys(smoke_artifacts):
    payload = json.loads(smoke_artifacts.phase1_report.read_text(encoding="utf-8"))
    must_have = {
        "reward_normalization_resolved",
        "dataset_reject_rate",
        "composite_score_sensitivity",
        "prospective_diagnostic_required",
        "stratification_mode",
    }
    assert must_have.issubset(payload.keys())


def test_input_schema_excludes_close(smoke_artifacts):
    schema = json.loads(smoke_artifacts.input_schema_json.read_text(encoding="utf-8"))
    assert schema["price_column"] == "close"
    assert "close" not in schema["feature_columns"]


def test_horizon_labels_within_archetype_range(smoke_artifacts):
    import polars as pl

    labels = pl.read_ipc(smoke_artifacts.horizon_labels_val)
    if labels.height > 0:
        max_id = labels["code_label"].max()
        assert 0 <= max_id <= 3  # K=4
