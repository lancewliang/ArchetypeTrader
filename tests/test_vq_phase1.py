"""VQ Phase I 单元测试

# 验证目标:
# 1. Section 4.1 / Eq. 2: 最近邻量化索引必须正确。
# 2. Section 4.1 / Eq. 4: encoder / codebook / decoder 梯度流必须正常。
# 3. 在极小数据集上，VQ 模型应能明显过拟合，证明实现链路可学习。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.phase1.codebook import VQCodebook
from src.phase1.vq_decoder import VQDecoder
from src.phase1.vq_encoder import VQEncoder



def test_codebook_quantize_returns_correct_nearest_neighbor() -> None:
    """验证 Eq. 2 的最近邻量化结果正确。"""
    codebook = VQCodebook(num_codes=3, code_dim=2)
    with torch.no_grad():
        codebook.embeddings.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [2.0, 2.0],
                    [-2.0, -2.0],
                ],
                dtype=torch.float32,
            )
        )

    z_e = torch.tensor(
        [
            [0.1, -0.1],
            [1.7, 2.2],
            [-1.5, -1.8],
        ],
        dtype=torch.float32,
    )
    _, indices, _ = codebook.quantize(z_e)

    assert indices.tolist() == [0, 1, 2]



def test_gradients_flow_through_vq_stack() -> None:
    """验证 encoder / codebook / decoder 在联合损失下都能拿到梯度。"""
    torch.manual_seed(7)

    encoder = VQEncoder(state_dim=4, action_dim=3, hidden_dim=8, latent_dim=3)
    codebook = VQCodebook(num_codes=4, code_dim=3)
    decoder = VQDecoder(state_dim=4, code_dim=3, hidden_dim=8, action_dim=3)

    s_demo = torch.randn(5, 6, 4)
    a_demo = torch.randint(low=0, high=3, size=(5, 6))
    r_demo = torch.randn(5, 6)

    z_e = encoder(s_demo, a_demo, r_demo)
    z_q_st, _, commitment_loss = codebook.quantize(z_e)
    logits = decoder(s_demo, z_q_st)

    ce_loss = nn.CrossEntropyLoss()(logits.reshape(-1, 3), a_demo.reshape(-1))
    encoder_commitment = 0.25 * torch.mean((z_e - z_q_st.detach()) ** 2)
    total_loss = ce_loss + commitment_loss + encoder_commitment
    total_loss.backward()

    encoder_grad = sum(
        float(p.grad.abs().sum()) for p in encoder.parameters() if p.grad is not None
    )
    codebook_grad = sum(
        float(p.grad.abs().sum()) for p in codebook.parameters() if p.grad is not None
    )
    decoder_grad = sum(
        float(p.grad.abs().sum()) for p in decoder.parameters() if p.grad is not None
    )

    assert encoder_grad > 0.0
    assert codebook_grad > 0.0
    assert decoder_grad > 0.0



def test_vq_stack_can_overfit_tiny_dataset() -> None:
    """验证极小数据集上的可学习性。

    这里不是追求论文级性能，而是要证明实现链路没有形状、梯度、量化接入错误。
    """
    torch.manual_seed(11)
    np.random.seed(11)

    batch_size = 16
    horizon = 6
    state_dim = 4
    action_dim = 3

    # 造一个极简但可学习的小数据集：
    # pattern A: 前半段 flat，后半段 long
    # pattern B: 前半段 flat，后半段 short
    s_demo = torch.randn(batch_size, horizon, state_dim)
    r_demo = torch.zeros(batch_size, horizon)
    a_demo = torch.ones(batch_size, horizon, dtype=torch.long)
    a_demo[: batch_size // 2, horizon // 2 :] = 2
    a_demo[batch_size // 2 :, horizon // 2 :] = 0

    encoder = VQEncoder(state_dim=state_dim, action_dim=action_dim, hidden_dim=16, latent_dim=8)
    codebook = VQCodebook(num_codes=4, code_dim=8)
    decoder = VQDecoder(state_dim=state_dim, code_dim=8, hidden_dim=16, action_dim=action_dim)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(codebook.parameters()) + list(decoder.parameters()),
        lr=1e-2,
    )
    ce_loss_fn = nn.CrossEntropyLoss()

    for _ in range(160):
        z_e = encoder(s_demo, a_demo, r_demo)
        z_q_st, _, commitment_loss = codebook.quantize(z_e)
        logits = decoder(s_demo, z_q_st)
        rec_loss = ce_loss_fn(logits.reshape(-1, action_dim), a_demo.reshape(-1))
        encoder_commitment = 0.25 * torch.mean((z_e - z_q_st.detach()) ** 2)
        loss = rec_loss + commitment_loss + encoder_commitment

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        z_e = encoder(s_demo, a_demo, r_demo)
        z_q_st, _, _ = codebook.quantize(z_e)
        logits = decoder(s_demo, z_q_st)
        pred = torch.argmax(logits, dim=-1)
        step_acc = float((pred == a_demo).float().mean().item())

    assert step_acc >= 0.95
