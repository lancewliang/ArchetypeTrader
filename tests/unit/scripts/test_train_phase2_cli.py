"""``scripts.train_phase2`` CLI tests."""
from __future__ import annotations

from scripts.train_phase2 import build_config, build_parser


def _required_args(tmp_path, *extra):
    return [
        "--pair",
        "FU",
        "--phase1-batch-id",
        "batch_p1",
        "--phase2-batch-id",
        "batch_p2",
        "--train-file",
        "data/FU/train.feather",
        "--val-file",
        "data/FU/val.feather",
        "--test-file",
        "data/FU/test.feather",
        "--artifact-root",
        str(tmp_path / "artifacts"),
        *extra,
    ]


def test_train_phase2_cli_inherits_phase1_max_position(tmp_path):
    p1_dir = tmp_path / "artifacts" / "FU" / "batch_p1" / "phase1"
    p1_dir.mkdir(parents=True)
    (p1_dir / "phase1_config.yaml").write_text(
        "dp:\n  max_position: 10\n",
        encoding="utf-8",
    )

    args = build_parser().parse_args(_required_args(tmp_path))
    config = build_config(args)

    assert config.max_position == 10


def test_train_phase2_cli_explicit_max_position_wins(tmp_path):
    p1_dir = tmp_path / "artifacts" / "FU" / "batch_p1" / "phase1"
    p1_dir.mkdir(parents=True)
    (p1_dir / "phase1_config.yaml").write_text(
        "dp:\n  max_position: 10\n",
        encoding="utf-8",
    )

    args = build_parser().parse_args(_required_args(tmp_path, "--max-position", "3"))
    config = build_config(args)

    assert config.max_position == 3
