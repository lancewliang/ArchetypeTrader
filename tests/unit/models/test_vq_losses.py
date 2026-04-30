"""``Phase1Loss`` 单元测试."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.vq_losses import Phase1Loss


def _logits_and_targets(near_perfect: bool = True):
    targets = torch.tensor([[1, 1, 2, 0]])
    logits = torch.zeros(1, 4, 3)
    if near_perfect:
        for t in range(4):
            logits[0, t, int(targets[0, t])] = 10.0
    return logits, targets


def test_perfect_logits_low_reconstruction_loss():
    logits, targets = _logits_and_targets(True)
    z_e = torch.randn(1, 8)
    z_q = z_e.clone()
    out = Phase1Loss(beta0=0.25, usage_weight=0.0)(
        action_logits=logits,
        target_actions=targets,
        z_e=z_e,
        z_q_no_grad=z_q,
        code_id=torch.tensor([0]),
    )
    assert out.reconstruction.item() < 0.1


def test_total_includes_codebook_and_commitment():
    logits, targets = _logits_and_targets(True)
    z_e = torch.randn(2, 8, requires_grad=True)
    z_q = z_e.detach() + 0.5  # 让 codebook 与 commitment 都有非零梯度路径
    out = Phase1Loss(beta0=0.25, usage_weight=0.0)(
        action_logits=logits.expand(2, -1, -1).contiguous(),
        target_actions=targets.expand(2, -1).contiguous(),
        z_e=z_e,
        z_q_no_grad=z_q,
        code_id=torch.tensor([0, 1]),
    )
    expected = out.reconstruction + out.codebook + 0.25 * out.commitment
    assert torch.allclose(out.total, expected)


def test_usage_loss_zero_when_weight_zero():
    logits, targets = _logits_and_targets(True)
    z_e = torch.randn(1, 8)
    z_q = z_e.clone()
    out = Phase1Loss(usage_weight=0.0)(
        action_logits=logits,
        target_actions=targets,
        z_e=z_e,
        z_q_no_grad=z_q,
        code_id=torch.tensor([0]),
    )
    assert out.usage is None


def test_contrastive_zero_when_no_pairs():
    logits, targets = _logits_and_targets(True)
    z_e = torch.randn(1, 8)
    z_q = z_e.clone()
    out = Phase1Loss(contrastive_weight=0.05)(
        action_logits=logits,
        target_actions=targets,
        z_e=z_e,
        z_q_no_grad=z_q,
        code_id=torch.tensor([0]),
        contrastive_pair_ids=None,
    )
    assert out.contrastive is None


def test_contrastive_pulls_pair_close():
    z_e = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    targets = torch.zeros(2, 2, dtype=torch.long)
    logits = torch.zeros(2, 2, 3)
    pair_ids = ["p", "p"]
    out = Phase1Loss(contrastive_weight=1.0)(
        action_logits=logits,
        target_actions=targets,
        z_e=z_e,
        z_q_no_grad=z_e.detach(),
        code_id=torch.tensor([0, 1]),
        contrastive_pair_ids=pair_ids,
    )
    # cosine 距离 > 0；loss 应当大于 0
    assert out.contrastive is not None and out.contrastive.item() > 0.0
