"""``LatentVisualizationWriter`` 单元测试."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.phase1.evaluation.diagnostics.latent_visualization import (
    LatentSnapshot,
    LatentVisualizationWriter,
)


def test_should_capture_at_first_epoch(tmp_path):
    writer = LatentVisualizationWriter(
        log_dir=tmp_path,
        log_every_epochs=5,
        max_points_per_split=10,
        fixed_probe_seed=2026,
        projections=("pca",),
    )
    assert writer.should_capture(0)


def test_should_skip_intermediate_epoch(tmp_path):
    writer = LatentVisualizationWriter(
        log_dir=tmp_path,
        log_every_epochs=5,
        max_points_per_split=10,
        fixed_probe_seed=2026,
        projections=("pca",),
    )
    assert not writer.should_capture(2)


def test_should_capture_at_log_interval(tmp_path):
    writer = LatentVisualizationWriter(
        log_dir=tmp_path,
        log_every_epochs=5,
        max_points_per_split=10,
        fixed_probe_seed=2026,
        projections=("pca",),
    )
    assert writer.should_capture(5)


def test_pca_projection_dim_2(tmp_path):
    np = pytest.importorskip("numpy")
    writer = LatentVisualizationWriter(
        log_dir=tmp_path,
        log_every_epochs=5,
        max_points_per_split=100,
        fixed_probe_seed=2026,
        projections=("pca",),
    )
    snap = LatentSnapshot(epoch=0, z_e=np.random.randn(20, 8).tolist())
    out = writer.compute_projections(snap, methods=("pca",))
    assert "pca" in out
    assert len(out["pca"][0]) == 2


def test_write_manifest_json(tmp_path):
    writer = LatentVisualizationWriter(
        log_dir=tmp_path,
        log_every_epochs=5,
        max_points_per_split=10,
        fixed_probe_seed=2026,
        projections=("pca",),
    )
    target = tmp_path / "manifest.json"
    writer.write_manifest(target, {"a": 1})
    assert target.exists()
