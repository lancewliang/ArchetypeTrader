"""``scripts.process_phase1_data`` tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.process_phase1_data import (
    Phase1DataProcessor,
    assert_prospective_diagnostic,
    build_data_process_config,
    build_parser,
)
from tests.fixtures.phase1.build_fixtures import FixtureSpec, build_fixtures


def _args(tmp_path: Path, *, data_batch_id: str = "processed"):
    fixtures_dir = tmp_path / f"fixtures_{data_batch_id}"
    train, val, test = build_fixtures(
        fixtures_dir,
        FixtureSpec(train_rows=96, val_rows=48, test_rows=48, seed=123),
    )
    factor_file = tmp_path / "factors" / "TEST" / "short.txt"
    factor_file.parent.mkdir(parents=True, exist_ok=True)
    factor_file.write_text("mid_price\nreturn_1m\n", encoding="utf-8")
    parser = build_parser()
    return parser.parse_args(
        [
            "--pair",
            "TEST",
            "--data-batch-id",
            data_batch_id,
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
            "4",
            "--stratification-mode",
            "prospective_past",
            "--local-smoke-relaxed-guardrails",
            "--seed",
            "11",
        ]
    )


def test_phase1_data_processor_writes_manifest_and_split_files(tmp_path):
    config = build_data_process_config(_args(tmp_path))

    manifest_path = Phase1DataProcessor(config).run()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pair"] == "TEST"
    for split in ("train", "val", "test"):
        split_payload = manifest["splits"][split]
        assert (manifest_path.parent / split_payload["sampled_horizons_path"]).exists()
        assert (manifest_path.parent / split_payload["dp_teacher_path"]).exists()
        assert (manifest_path.parent / split_payload["window_index_path"]).exists()


def test_phase1_data_processor_records_schema_and_hashes(tmp_path):
    config = build_data_process_config(_args(tmp_path))

    manifest_path = Phase1DataProcessor(config).run()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_hash"]
    assert manifest["data_process_hash"]
    assert manifest["dp_teacher_hash"]
    assert manifest["feature_provenance_path"] == "feature_provenance.json"
    provenance_path = manifest_path.parent / manifest["feature_provenance_path"]
    assert provenance_path.exists()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert set(provenance["feature_columns"]) == set(manifest["feature_source"]["feature_columns"])
    assert manifest["feature_source"]["mode"] == "fixed_plus_factor_list"


def test_phase1_data_processor_writes_val_and_test_teacher(tmp_path):
    config = build_data_process_config(_args(tmp_path))
    manifest_path = Phase1DataProcessor(config).run()
    import polars as pl

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in ("val", "test"):
        teacher = pl.read_ipc(manifest_path.parent / manifest["splits"][split]["dp_teacher_path"])
        assert teacher.height == manifest["splits"][split]["num_horizons"]
        assert "actions" in teacher.columns
        assert "rewards" in teacher.columns


def test_phase1_data_processor_labels_eval_splits_all_boundary_eligible(tmp_path):
    config = build_data_process_config(_args(tmp_path))
    manifest_path = Phase1DataProcessor(config).run()
    import polars as pl

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in ("val", "test"):
        split_payload = manifest["splits"][split]
        window_index = pl.read_ipc(
            manifest_path.parent / split_payload["window_index_path"]
        )
        teacher = pl.read_ipc(
            manifest_path.parent / split_payload["dp_teacher_path"]
        )
        eligible_count = window_index.filter(pl.col("is_boundary_eligible")).height
        assert teacher.height == eligible_count
        assert split_payload["num_labeled_windows"] == eligible_count
        assert split_payload["labeling_mode"] == "all_eligible"
        assert split_payload["sampling_applied"] is False


def test_phase1_data_processor_records_train_sample_source(tmp_path):
    config = build_data_process_config(_args(tmp_path))
    manifest_path = Phase1DataProcessor(config).run()
    import polars as pl

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_payload = manifest["splits"]["train"]
    sampled = pl.read_ipc(manifest_path.parent / split_payload["sampled_horizons_path"])
    assert "sample_source" in sampled.columns
    assert "full_time" in set(sampled["sample_source"].to_list())
    assert split_payload["sample_source_counts"]["full_time"] >= 1
    assert "coverage_after_dp" in split_payload


def test_phase1_data_processor_preserves_deterministic_sample_ids(tmp_path):
    first = build_data_process_config(_args(tmp_path, data_batch_id="p1"))
    second = build_data_process_config(_args(tmp_path, data_batch_id="p2"))

    first_manifest = Phase1DataProcessor(first).run()
    second_manifest = Phase1DataProcessor(second).run()

    import polars as pl

    first_payload = json.loads(first_manifest.read_text(encoding="utf-8"))
    second_payload = json.loads(second_manifest.read_text(encoding="utf-8"))
    first_samples = pl.read_ipc(
        first_manifest.parent / first_payload["splits"]["train"]["sampled_horizons_path"]
    )["sample_id"].to_list()
    second_samples = pl.read_ipc(
        second_manifest.parent / second_payload["splits"]["train"]["sampled_horizons_path"]
    )["sample_id"].to_list()
    assert first_samples == second_samples


def test_phase1_data_processor_rejects_missing_prospective_diagnostic(tmp_path):
    args = _args(tmp_path)
    args.stratification_mode = "hindsight_horizon"
    args.diagnostic_pair_batch_id = None
    args.allow_missing_prospective_diagnostic = False

    with pytest.raises(SystemExit):
        assert_prospective_diagnostic(args)
