#!/usr/bin/env python
"""Phase I 训练脚本 — 原型发现

# 需求: 7.1, 4.6, 4.7, 4.8, 7.5, 7.6, 7.7
#
# 流程:
# 1. 加载特征数据，初始化 TradingEnv
# 2. 调用 DPPlanner 生成 30k 示范轨迹并保存
# 3. 初始化 VQ Encoder、Codebook、Decoder
# 4. 训练 100 epochs
#    损失函数 L = L_rec + ||sg[z_e] - z_q||² + 0.25 × ||z_e - sg[z_q]||²
# 5. 保存模型到 result/phase1_archetype_discovery/
# 6. 输出训练日志（loss 曲线）
#
# 用法:
#   python scripts/train_phase1.py --pair BTC
#   python scripts/train_phase1.py --pair ETH --phase1-epochs 50 --batch-size 128
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import parse_args
from src.data.dataset import TrajectoryDataset
from src.data.feature_pipeline import FeaturePipeline
from src.env.trading_env import TradingEnv
from src.phase1.codebook import VQCodebook
from src.phase1.dp_planner import DPPlanner
from src.phase1.vq_decoder import VQDecoder
from src.phase1.vq_encoder import VQEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    # ----------------------------------------------------------------
    # Step 0: 解析配置
    # ----------------------------------------------------------------
    config = parse_args()
    pair = config.pairs[0]  # 单交易对训练
    logger.info("Phase I 训练开始: pair=%s", pair)
    logger.info(
        "超参数: epochs=%d, batch_size=%d, lr=%.1e, latent_dim=%d, "
        "num_archetypes=%d, num_trajectories=%d, vq_beta0=%.2f",
        config.phase1_epochs,
        config.batch_size,
        config.learning_rate,
        config.latent_dim,
        config.num_archetypes,
        config.num_trajectories,
        config.vq_beta0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    # ----------------------------------------------------------------
    # Step 1: 加载特征数据，初始化 TradingEnv
    # ----------------------------------------------------------------
    logger.info("加载特征数据: data_dir=%s, pair=%s", config.data_dir, pair)
    pipeline = FeaturePipeline(config.data_dir, pair, config)
    states = pipeline.get_state_vector()  # (T, 45)

    # 使用第一列作为价格代理（wap 特征）
    # [NOTE: 论文未明确指定价格列，使用 states 第 0 列作为价格]
    prices = states[:, 0].copy()

    # 按时间划分，仅使用训练集
    train_states, _, _ = pipeline.split_by_date(states)
    train_prices = prices[: len(train_states)]

    logger.info(
        "训练集: states shape=%s, prices shape=%s",
        train_states.shape,
        train_prices.shape,
    )

    env = TradingEnv(
        states=train_states,
        prices=train_prices,
        pair=pair,
        horizon=config.horizon,
    )
    logger.info("TradingEnv 初始化完成: num_horizons=%d", env.num_horizons)

    # ----------------------------------------------------------------
    # Step 2: 生成 DP 示范轨迹
    # ----------------------------------------------------------------
    traj_path = os.path.join(
        config.result_dir, "dp_trajectories", f"{pair}_trajectories.npz"
    )

    if os.path.exists(traj_path):
        logger.info("发现已有轨迹文件，直接加载: %s", traj_path)
        dataset = TrajectoryDataset.from_npz(traj_path)
    else:
        logger.info(
            "开始生成 DP 示范轨迹: num_trajectories=%d",
            config.num_trajectories,
        )
        planner = DPPlanner(env)
        trajectories = planner.generate_trajectories(config.num_trajectories)
        logger.info("DP 轨迹生成完成，创建 Dataset")
        dataset = TrajectoryDataset(
            states=trajectories["states"],
            actions=trajectories["actions"],
            rewards=trajectories["rewards"],
        )

    logger.info("Dataset 大小: %d 条轨迹", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # ----------------------------------------------------------------
    # Step 3: 初始化 VQ Encoder、Codebook、Decoder
    # ----------------------------------------------------------------
    encoder = VQEncoder(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.lstm_hidden_dim,
        latent_dim=config.latent_dim,
    ).to(device)

    codebook = VQCodebook(
        num_codes=config.num_archetypes,
        code_dim=config.latent_dim,
    ).to(device)

    decoder = VQDecoder(
        state_dim=config.state_dim,
        code_dim=config.latent_dim,
        hidden_dim=config.lstm_hidden_dim,
        action_dim=config.action_dim,
    ).to(device)

    logger.info(
        "模型初始化完成: Encoder params=%d, Codebook params=%d, Decoder params=%d",
        sum(p.numel() for p in encoder.parameters()),
        sum(p.numel() for p in codebook.parameters()),
        sum(p.numel() for p in decoder.parameters()),
    )

    # 优化器：联合训练 encoder + codebook + decoder
    all_params = (
        list(encoder.parameters())
        + list(codebook.parameters())
        + list(decoder.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=config.learning_rate)

    # 重建损失：交叉熵（decoder 输出 logits vs ground-truth actions）
    ce_loss_fn = nn.CrossEntropyLoss()

    # ----------------------------------------------------------------
    # Step 4: 训练循环 — 100 epochs
    # 损失函数 L = L_rec + ||sg[z_e] - z_q||² + β₀ × ||z_e - sg[z_q]||²
    # VQCodebook.quantize() 返回的 commitment_loss = ||sg[z_e] - z_q||²
    # β₀ × ||z_e - sg[z_q]||² 需要在训练循环中额外计算
    # ----------------------------------------------------------------
    loss_history = []

    logger.info("开始训练: %d epochs", config.phase1_epochs)

    for epoch in range(1, config.phase1_epochs + 1):
        encoder.train()
        codebook.train()
        decoder.train()

        epoch_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_vq_loss = 0.0
        num_batches = 0

        for s_demo, a_demo, r_demo in dataloader:
            s_demo = s_demo.to(device)   # (B, h, state_dim)
            a_demo = a_demo.to(device)   # (B, h)
            r_demo = r_demo.to(device)   # (B, h)

            # Phase I, Step 1: Encode
            z_e = encoder(s_demo, a_demo, r_demo)  # (B, latent_dim)

            # Phase I, Step 2: Quantize
            z_q_st, indices, commitment_loss = codebook.quantize(z_e)
            # z_q_st: straight-through (B, latent_dim)
            # commitment_loss: ||sg[z_e] - z_q||²

            # Phase I, Step 3: Decode
            action_logits = decoder(s_demo, z_q_st)  # (B, h, action_dim)

            # L_rec: 交叉熵重建损失
            # reshape for cross-entropy: (B*h, action_dim) vs (B*h,)
            logits_flat = action_logits.reshape(-1, config.action_dim)
            targets_flat = a_demo.reshape(-1)
            rec_loss = ce_loss_fn(logits_flat, targets_flat)

            # β₀ × ||z_e - sg[z_q]||²
            z_q_detached = z_q_st.detach()  # sg[z_q] — stop gradient on z_q
            encoder_commitment = config.vq_beta0 * torch.mean(
                (z_e - z_q_detached) ** 2
            )

            # 总损失: L = L_rec + commitment_loss + β₀ × encoder_commitment
            # = L_rec + ||sg[z_e] - z_q||² + 0.25 × ||z_e - sg[z_q]||²
            total_loss = rec_loss + commitment_loss + encoder_commitment

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_rec_loss += rec_loss.item()
            epoch_vq_loss += (commitment_loss.item() + encoder_commitment.item())
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_rec = epoch_rec_loss / max(num_batches, 1)
        avg_vq = epoch_vq_loss / max(num_batches, 1)
        loss_history.append(avg_loss)

        # 每 10 个 epoch 或首尾 epoch 输出日志
        if epoch == 1 or epoch % 10 == 0 or epoch == config.phase1_epochs:
            logger.info(
                "Epoch %3d/%d — total_loss=%.4f, rec_loss=%.4f, vq_loss=%.4f",
                epoch,
                config.phase1_epochs,
                avg_loss,
                avg_rec,
                avg_vq,
            )

        # NaN 检测
        if np.isnan(avg_loss):
            logger.error("训练 loss 发散 (NaN)，在 epoch %d 终止训练", epoch)
            break

    # ----------------------------------------------------------------
    # Step 5: 保存模型到 result/phase1_archetype_discovery/
    # ----------------------------------------------------------------
    save_dir = os.path.join(config.result_dir, "phase1_archetype_discovery")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{pair}_vq_model.pt")
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "codebook": codebook.state_dict(),
            "decoder": decoder.state_dict(),
            "loss_history": loss_history,
            "config": {
                "state_dim": config.state_dim,
                "action_dim": config.action_dim,
                "latent_dim": config.latent_dim,
                "num_archetypes": config.num_archetypes,
                "lstm_hidden_dim": config.lstm_hidden_dim,
                "phase1_epochs": config.phase1_epochs,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "vq_beta0": config.vq_beta0,
            },
        },
        save_path,
    )
    logger.info("模型已保存到 %s", save_path)

    # ----------------------------------------------------------------
    # Step 6: 输出训练日志摘要
    # ----------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("Phase I 训练完成: pair=%s", pair)
    logger.info("最终 loss: %.4f", loss_history[-1] if loss_history else float("nan"))
    logger.info("最低 loss: %.4f (epoch %d)",
                min(loss_history) if loss_history else float("nan"),
                (loss_history.index(min(loss_history)) + 1) if loss_history else 0)
    logger.info("模型保存路径: %s", save_path)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
