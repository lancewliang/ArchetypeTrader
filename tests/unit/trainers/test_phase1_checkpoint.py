"""``Phase1CheckpointManager`` 单元测试."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.trainers.phase1_checkpoint import (
    Phase1CheckpointManager,
    Phase1FatalCollapse,
    file_sha256,
)
from src.trainers.selection_policy import SelectionVerdict


def _state():
    return {"model": {"w": torch.zeros(3)}, "epoch": 0}


def test_save_last_creates_file(tmp_path):
    mgr = Phase1CheckpointManager(tmp_path)
    out = mgr.save_last(_state(), {"loss": 0.1}, epoch=0)
    assert out.exists()


def test_save_periodic_only_when_due(tmp_path):
    mgr = Phase1CheckpointManager(tmp_path)
    p = mgr.save_periodic(_state(), {"loss": 0.1}, epoch=4, save_every=5)
    assert p is None
    p = mgr.save_periodic(_state(), {"loss": 0.1}, epoch=5, save_every=5)
    assert p is not None and p.exists()


def test_promote_to_best_copies_last(tmp_path):
    mgr = Phase1CheckpointManager(tmp_path)
    mgr.save_last(_state(), {"loss": 0.1}, epoch=0)
    verdict = SelectionVerdict(decision="promote_to_best", composite_score=0.5)
    entry = mgr.commit_verdict(_state(), {"loss": 0.1}, verdict, epoch=0)
    assert entry.is_best
    assert mgr.best_path.exists()
    # best 与 last 内容应该一致
    assert mgr.best_path.read_bytes() == mgr.last_path.read_bytes()


def test_reject_keeps_best_unchanged(tmp_path):
    mgr = Phase1CheckpointManager(tmp_path)
    mgr.save_last(_state(), {"loss": 0.1}, epoch=0)
    # 第一次 promote
    mgr.commit_verdict(_state(), {"loss": 0.1}, SelectionVerdict("promote_to_best", composite_score=1.0), 0)
    best_hash = file_sha256(mgr.best_path)
    # 后续 reject
    mgr.save_last({"model": {"w": torch.ones(3)}, "epoch": 1}, {"loss": 0.5}, epoch=1)
    mgr.commit_verdict(
        {"model": {"w": torch.ones(3)}, "epoch": 1},
        {"loss": 0.5},
        SelectionVerdict("reject", reasons=["risk_guardrail"]),
        epoch=1,
    )
    assert file_sha256(mgr.best_path) == best_hash


def test_fatal_raises_phase1_fatal_collapse(tmp_path):
    mgr = Phase1CheckpointManager(tmp_path)
    mgr.save_last(_state(), {"loss": 0.1}, epoch=0)
    verdict = SelectionVerdict("fatal", reasons=["consecutive_collapse"])
    with pytest.raises(Phase1FatalCollapse):
        mgr.commit_verdict(_state(), {"loss": 0.1}, verdict, epoch=0)


def test_manifest_records_verdict(tmp_path):
    mgr = Phase1CheckpointManager(tmp_path)
    mgr.save_last(_state(), {"loss": 0.1}, epoch=0)
    mgr.commit_verdict(
        _state(), {"loss": 0.1}, SelectionVerdict("promote_to_best", composite_score=1.0), 0,
    )
    manifest = json.loads(mgr.manifest_path.read_text(encoding="utf-8"))
    assert manifest[0]["verdict"] == "promote_to_best"


def test_file_sha256_changes_when_content_changes(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"world")
    assert file_sha256(a) != file_sha256(b)
