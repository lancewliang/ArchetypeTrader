"""Two-stage Phase I data-process then manifest-train smoke."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("polars")

from scripts.process_phase1_data import (  # noqa: E402
    Phase1DataProcessor,
    build_data_process_config,
    build_parser,
)
from src.config.phase1_config import (  # noqa: E402
    BehaviorGuardrailConfig,
    CodebookConfig,
    CodebookHealthConfig,
    CostConfig,
    DPConfig,
    DiagnosticsConfig,
    EncoderInputConfig,
    ModelConfig,
    Phase1Config,
    RiskGuardrailConfig,
    SamplingHealthConfig,
    SelectionPolicyConfig,
    StratificationConfig,
    TrainingConfig,
)
from src.data.horizon_builder import HorizonBuilder  # noqa: E402
from src.data.market_reader import MarketFileReader  # noqa: E402
from src.data.stratified_sampler import StratifiedWindowSampler  # noqa: E402
from src.data.window_indexer import SlidingWindowIndexer  # noqa: E402
from src.planners.demo_generator import Phase1DemoGenerator  # noqa: E402
from src.planners.single_trade_dp import SingleTradeDPPlanner  # noqa: E402
from src.trainers.phase1_trainer import Phase1Trainer  # noqa: E402
from tests.fixtures.phase1.build_fixtures import FixtureSpec, build_fixtures  # noqa: E402


def _processor_args(tmp_path: Path):
    train, val, test = build_fixtures(
        tmp_path / "fixtures",
        FixtureSpec(train_rows=160, val_rows=80, test_rows=80, seed=321),
    )
    factor_file = tmp_path / "factors" / "TEST" / "short.txt"
    factor_file.parent.mkdir(parents=True, exist_ok=True)
    factor_file.write_text("mid_price\nreturn_1m\n", encoding="utf-8")
    return build_parser().parse_args(
        [
            "--pair",
            "TEST",
            "--data-batch-id",
            "processed",
            "--train-file",
            str(train),
            "--val-file",
            str(val),
            "--test-file",
            str(test),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--factor-list-file",
            str(factor_file),
            "--horizon",
            "4",
            "--num-demos",
            "8",
            "--stratification-mode",
            "prospective_past",
            "--local-smoke-relaxed-guardrails",
            "--seed",
            "7",
        ]
    )


def _train_config(tmp_path: Path, manifest: Path) -> Phase1Config:
    cost = CostConfig(reward_alignment="paper_formula")
    dp = DPConfig(horizon=4, cost_config=cost, max_position=1, gamma=1.0)
    enc = EncoderInputConfig(
        state_adapter_dim=8,
        action_embedding_dim=4,
        reward_embedding_dim=4,
        fusion_dim=16,
    )
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
    model = ModelConfig(
        hidden_dim=16,
        code_dim=4,
        num_codes=4,
        encoder_input=enc,
        codebook=cb,
    )
    training = TrainingConfig(
        batch_size=4,
        lr=1e-3,
        epochs=1,
        seed=7,
        device="cpu",
        save_every=1,
        full_validation_every_epochs=1,
        fast_val_probe_size=8,
    )
    selection = SelectionPolicyConfig(
        min_code_usage_ratio=0.0,
        risk=RiskGuardrailConfig(max_drawdown=10.0, min_sharpe_ratio=-999.0),
        behavior=BehaviorGuardrailConfig(
            min_inter_code_action_diversity=0.0,
            min_decoder_sensitivity_to_code=0.0,
            min_epoch_code_stability=0.0,
        ),
    )
    return Phase1Config(
        pair="TEST",
        train_batch_id="manifest_train",
        data_process_manifest=str(manifest),
        artifact_root=str(tmp_path / "artifacts"),
        horizon=4,
        stratification=StratificationConfig(mode="prospective_past"),
        dp=dp,
        model=model,
        training=training,
        selection_policy=selection,
        diagnostics=DiagnosticsConfig(
            failure_cases_enabled=False,
            latent_visualization_enabled=False,
        ),
        local_smoke_relaxed_guardrails=True,
    )


def test_phase1_data_process_then_manifest_train_smoke(tmp_path, monkeypatch):
    manifest = Phase1DataProcessor(
        build_data_process_config(_processor_args(tmp_path))
    ).run()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("manifest training must not regenerate processed data")

    monkeypatch.setattr(MarketFileReader, "read_split", forbidden)
    monkeypatch.setattr(SlidingWindowIndexer, "enumerate", forbidden)
    monkeypatch.setattr(StratifiedWindowSampler, "sample", forbidden)
    monkeypatch.setattr(HorizonBuilder, "build", forbidden)
    monkeypatch.setattr(Phase1DemoGenerator, "generate", forbidden)
    monkeypatch.setattr(SingleTradeDPPlanner, "plan", forbidden)

    artifacts = Phase1Trainer(_train_config(tmp_path, manifest)).run()

    for path in (
        artifacts.best_vq_model,
        artifacts.horizon_labels_train,
        artifacts.horizon_labels_val,
        artifacts.horizon_labels_test,
        artifacts.phase1_report,
    ):
        assert path.exists()
    report = json.loads(artifacts.phase1_report.read_text(encoding="utf-8"))
    assert report["processed_data_mode"] == "manifest"
    assert report["data_process_manifest"] == str(manifest)
    assert report["data_process_hash"]
    assert report["dp_teacher_hash"]
