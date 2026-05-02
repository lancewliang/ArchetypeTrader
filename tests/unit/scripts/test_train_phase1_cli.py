"""``scripts.train_phase1`` CLI tests."""
from __future__ import annotations

import pytest

from scripts.train_phase1 import build_config, build_parser, main


def test_train_phase1_cli_manifest_mode_does_not_require_raw_files(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--pair",
            "TEST",
            "--train-batch-id",
            "run",
            "--data-process-manifest",
            str(tmp_path / "data_process_manifest.json"),
        ]
    )

    config = build_config(args)

    assert config.train_file == ""
    assert config.val_file == ""
    assert config.test_file == ""
    assert config.data_process_manifest.endswith("data_process_manifest.json")


def test_train_phase1_cli_legacy_mode_still_requires_raw_files():
    with pytest.raises(SystemExit):
        main(["--pair", "TEST", "--train-batch-id", "run"])
