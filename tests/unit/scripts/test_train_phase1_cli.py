"""``scripts.train_phase1`` CLI tests."""
from __future__ import annotations

import pytest

from scripts.train_phase1 import build_config, build_parser, main


def test_train_phase1_cli_manifest_mode_required():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--pair",
            "TEST",
            "--train-batch-id",
            "run",
            "--data-process-manifest",
            "/tmp/data_process_manifest.json",
        ]
    )

    config = build_config(args)

    assert config.data_process_manifest == "/tmp/data_process_manifest.json"


def test_train_phase1_cli_missing_manifest_exits():
    with pytest.raises(SystemExit):
        main(["--pair", "TEST", "--train-batch-id", "run"])
