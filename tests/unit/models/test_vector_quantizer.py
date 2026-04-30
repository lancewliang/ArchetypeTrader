"""``VectorQuantizer`` 单元测试."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.config.phase1_config import (
    CodebookConfig,
    CodebookHealthConfig,
    CodebookLocalOptimumEscapeConfig,
)
from src.models.vector_quantizer import VectorQuantizer


def _make_quantizer(num_codes: int = 4, code_dim: int = 4, update_method: str = "ema"):
    health = CodebookHealthConfig(
        usage_regularization_weight=0.0,
        dead_code_restart=True,
        local_optimum_escape=CodebookLocalOptimumEscapeConfig(),
    )
    cfg = CodebookConfig(
        init_method="random_normal",
        kmeans_warmup_batches=2,
        update_method=update_method,
        ema_decay=0.5,
        ema_epsilon=1e-5,
        health=health,
    )
    return VectorQuantizer(num_codes=num_codes, code_dim=code_dim, config=cfg)


def test_nearest_code_id_correct():
    vq = _make_quantizer()
    # 设置 codebook 为单位标识便于断言
    vq.codebook.data.zero_()
    for i in range(vq.num_codes):
        vq.codebook.data[i, i] = 1.0
    # 输入靠近 code 2
    z_e = torch.zeros(1, 4)
    z_e[0, 2] = 1.0
    out = vq.quantize(z_e)
    assert int(out.code_id.item()) == 2


def test_quantize_shape_and_grad_path():
    vq = _make_quantizer()
    z_e = torch.randn(8, 4, requires_grad=True)
    out = vq.quantize(z_e)
    assert out.z_q.shape == z_e.shape
    # STE: z_q.sum().backward() 应给 z_e 梯度（非 None）
    out.z_q.sum().backward()
    assert z_e.grad is not None


def test_usage_stats_basic():
    vq = _make_quantizer(num_codes=4)
    code_id = torch.tensor([0, 0, 0, 1, 1, 2])
    stats = vq.usage_stats(code_id)
    assert stats.code_usage_ratio == 3 / 4
    assert 3 in stats.dead_codes
    assert sum(stats.counts) == 6


def test_ema_update_changes_codebook():
    vq = _make_quantizer(update_method="ema")
    vq._warmup_initialized.fill_(True)
    vq.train()
    z_e = torch.randn(16, 4)
    out = vq.quantize(z_e)
    before = vq.codebook.data.clone()
    vq.update_codebook(z_e, out.code_id)
    after = vq.codebook.data
    assert not torch.allclose(before, after)


def test_gradient_mode_does_not_update_via_ema():
    vq = _make_quantizer(update_method="gradient")
    vq.train()
    before = vq.codebook.data.clone()
    z_e = torch.randn(16, 4)
    out = vq.quantize(z_e)
    vq.update_codebook(z_e, out.code_id)
    assert torch.allclose(before, vq.codebook.data)


def test_warmup_kmeans_changes_codebook_from_random():
    health = CodebookHealthConfig(usage_regularization_weight=0.0)
    cfg = CodebookConfig(
        init_method="kmeans_warmup",
        kmeans_warmup_batches=1,
        update_method="ema",
        health=health,
    )
    vq = VectorQuantizer(num_codes=4, code_dim=4, config=cfg)
    before = vq.codebook.data.clone()
    samples = torch.randn(64, 4)
    vq.warmup_initialize(samples)
    assert not torch.allclose(before, vq.codebook.data)
