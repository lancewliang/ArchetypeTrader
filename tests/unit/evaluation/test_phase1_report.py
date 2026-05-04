"""``Phase1ReportWriter`` 单元测试."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.phase1_report import (
    Phase1ReportWriter,
    REQUIRED_REPORT_KEYS,
    ReportPaths,
    ReportSchemaError,
)


def _make_summary(extra: dict | None = None):
    summary = {
        "reconstruction_accuracy": 0.5,
        "weighted_reconstruction_accuracy": 0.5,
        "non_flat_accuracy": 0.5,
        "code_usage": {"used": 7, "K": 10},
        "perplexity": 5.0,
        "single_trade_consistency_rate": 0.9,
        "no_trade_ratio": 0.1,
        "reward_alignment": "paper_formula",
        "reward_normalization_resolved": "train_reward_robust",
        "reward_norm_clip_ratio": 0.0,
        "dataset_reject_rate": 0.0,
        "processed_data_mode": "legacy_inline",
        "data_process_manifest": "",
        "data_batch_id": "",
        "schema_hash": "schema",
        "data_process_hash": "",
        "dp_teacher_hash": "",
        "stratification_mode": "hindsight_horizon",
        "is_hindsight_stratification": True,
        "prospective_diagnostic_required": True,
        "diagnostic_pair_batch_id": "batch_002",
        "phase1_composite_score": 0.0,
        "best_epoch": 0,
        "best_checkpoint_path": "best_vq_model.pt",
        "selection_metric": "phase1_composite_score",
        "composite_score_sensitivity": "composite_score_sensitivity.json",
        "best_checkpoint_signoff": True,
        "phase1_leakage_signoff": True,
        "phase1_checkpoint_eligible_for_phase2": True,
        "signoff_scope": "phase1_checkpoint_selection_and_no_leakage",
        "signoff_status": "passed",
        "signoff_blocked_reason": "",
        "signoff_blocking_reasons": [],
        "signoff_warning_reasons": [],
        "phase2_required_controls": [],
    }
    if extra:
        summary.update(extra)
    return summary


def test_required_keys_include_high_risk_fixes():
    keys = set(REQUIRED_REPORT_KEYS)
    must_have = {
        "reward_normalization_resolved",
        "dataset_reject_rate",
        "composite_score_sensitivity",
        "prospective_diagnostic_required",
        "signoff_status",
        "signoff_warning_reasons",
        "phase1_checkpoint_eligible_for_phase2",
    }
    assert must_have.issubset(keys)


def test_validate_schema_raises_when_missing(tmp_path):
    paths = ReportPaths.from_artifacts_dir(tmp_path)
    writer = Phase1ReportWriter(paths)
    with pytest.raises(ReportSchemaError):
        writer.validate_schema({"reconstruction_accuracy": 0.0})


def test_write_final_report_round_trip(tmp_path):
    paths = ReportPaths.from_artifacts_dir(tmp_path)
    writer = Phase1ReportWriter(paths)
    summary = _make_summary()
    out = writer.write_final_report(summary)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["selection_metric"] == "phase1_composite_score"


def test_unknown_keys_tolerated(tmp_path):
    paths = ReportPaths.from_artifacts_dir(tmp_path)
    writer = Phase1ReportWriter(paths)
    writer.validate_schema(_make_summary({"extra_key": 1}))


def test_write_diagnostics_writes_known_keys(tmp_path):
    paths = ReportPaths.from_artifacts_dir(tmp_path)
    writer = Phase1ReportWriter(paths)
    written = writer.write_diagnostics({"action": {"a": 1}, "risk": {"b": 2}})
    assert len(written) == 2
