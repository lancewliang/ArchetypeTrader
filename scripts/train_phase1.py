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
# 6. 执行 Phase I 验证并保存 phase1_validation_report.json
#
# 用法:
#   python scripts/train_phase1.py --pair BTC
#   python scripts/train_phase1.py --pair ETH --phase1-epochs 50 --batch-size 128
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import parse_args
from src.data.dataset import TrajectoryDataset
from src.data.feature_pipeline import FeaturePipeline
from src.env.trading_env import TradingEnv
from src.phase1.codebook import VQCodebook
from src.phase1.dp_planner import DPPlanner
from src.phase1.validation import validate_phase1_artifacts
from src.phase1.vq_decoder import VQDecoder
from src.phase1.vq_encoder import VQEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_reproducibility_seed(seed: int) -> None:
    """设置 Phase I 复现实验所需的随机种子。

    新增该方法是为了把 DP 采样、PyTorch 初始化与 DataLoader shuffle
    尽量固定下来，减少多次运行间的非必要波动。

    Args:
        seed: 随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_grad_norm(parameters) -> float:
    """计算参数梯度的全局 L2 范数。

    新增该方法用于训练监控，帮助后续定位梯度爆炸、梯度消失和模块失衡问题。
    """
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        norm = param.grad.detach().data.norm(2).item()
        total += norm * norm
    return float(total ** 0.5)


def summarize_code_usage(code_counts: np.ndarray) -> dict:
    """根据 epoch 内的 code 使用计数汇总 perplexity 与塌缩指标。"""
    total = int(np.sum(code_counts))
    if total <= 0:
        return {
            "used_code_count": 0,
            "dead_code_count": int(len(code_counts)),
            "dominant_code_ratio": 0.0,
            "codebook_entropy": 0.0,
            "codebook_perplexity": 1.0,
        }

    probs = code_counts.astype(np.float64) / float(total)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs))) if probs.size else 0.0
    perplexity = float(np.exp(entropy)) if entropy > 0 else 1.0
    return {
        "used_code_count": int(np.sum(code_counts > 0)),
        "dead_code_count": int(np.sum(code_counts == 0)),
        "dominant_code_ratio": float(np.max(code_counts) / total),
        "codebook_entropy": entropy,
        "codebook_perplexity": perplexity,
    }


def main() -> None:
    # ----------------------------------------------------------------
    # Step 0: 解析配置
    # ----------------------------------------------------------------
    config = parse_args()
    pair = config.pairs[0]  # 单交易对训练
    set_reproducibility_seed(config.phase1_sampling_seed)

    logger.info("Phase I 训练开始: pair=%s", pair)
    logger.info(
        "超参数: epochs=%d, batch_size=%d, lr=%.1e, latent_dim=%d, "
        "num_archetypes=%d, num_trajectories=%d, vq_beta0=%.2f, sampling_seed=%d",
        config.phase1_epochs,
        config.batch_size,
        config.learning_rate,
        config.latent_dim,
        config.num_archetypes,
        config.num_trajectories,
        config.vq_beta0,
        config.phase1_sampling_seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    # ----------------------------------------------------------------
    # Step 1: 加载特征数据，初始化 TradingEnv
    # ----------------------------------------------------------------
    logger.info("加载特征数据: data_dir=%s, pair=%s", config.data_dir, pair)
    pipeline = FeaturePipeline(config.data_dir, pair)
    train_df, _, _ = pipeline.get_state_vector()
    train_prices_df, _, _ = pipeline.get_prices()

    train_states = train_df.to_numpy()
    prices = train_prices_df["close"].to_numpy()

    logger.info(
        "训练集: states shape=%s, prices shape=%s",
        train_states.shape,
        prices.shape,
    )

    env = TradingEnv(
        states=train_states,
        prices=prices,
        pair=pair,
        horizon=config.horizon,
        states_dataframe=train_df,
        max_positions=config.max_positions,
        commission_rate=config.commission_rate,
    )

    logger.info(
        "TradingEnv 初始化完成 num_horizons = 总行数/切片内行数 = train_states.shape[0]/horizon: "
        "num_horizons=%d, horizon=%d, max_position=%d, commission_rate=%.6f",
        env.num_horizons,
        config.horizon,
        env.m,
        env.commission_rate,
    )

    # ----------------------------------------------------------------
    # Step 2: 生成 DP 示范轨迹
    # ----------------------------------------------------------------
    planner = DPPlanner(
        env=env,
        gamma=config.discount_factor,
        result_dir=config.result_dir,
        sampling_seed=config.phase1_sampling_seed,
    )
    traj_path = DPPlanner.build_trajectory_cache_path(config.result_dir, pair)

    if os.path.exists(traj_path):
        logger.info("发现已有轨迹文件，直接加载: %s", traj_path)
        dataset = TrajectoryDataset.from_npz(traj_path)
    else:
        logger.info(
            "开始生成 DP 示范轨迹: num_trajectories=%d",
            config.num_trajectories,
        )
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
    rec_loss_history = []
    vq_loss_history = []
    token_accuracy_history = []
    exact_match_history = []
    codebook_perplexity_history = []
    used_code_count_history = []
    dominant_code_ratio_history = []
    encoder_grad_norm_history = []
    codebook_grad_norm_history = []
    decoder_grad_norm_history = []
    logit_abs_max_history = []
    z_e_norm_history = []
    quantization_mse_history = []

    logger.info("开始训练: %d epochs", config.phase1_epochs)

    for epoch in tqdm(range(1, config.phase1_epochs + 1), desc="Training Epochs"):
        encoder.train()
        codebook.train()
        decoder.train()

        epoch_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_token_correct = 0
        epoch_token_total = 0
        epoch_exact_match = 0
        epoch_sample_total = 0
        epoch_code_counts = np.zeros(config.num_archetypes, dtype=np.int64)
        epoch_encoder_grad = 0.0
        epoch_codebook_grad = 0.0
        epoch_decoder_grad = 0.0
        epoch_logit_abs_max = 0.0
        epoch_z_e_norm_sum = 0.0
        epoch_quantization_mse_sum = 0.0
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
            pred_actions = torch.argmax(action_logits, dim=-1)

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

            epoch_encoder_grad += compute_grad_norm(encoder.parameters())
            epoch_codebook_grad += compute_grad_norm(codebook.parameters())
            epoch_decoder_grad += compute_grad_norm(decoder.parameters())

            optimizer.step()

            batch_size = int(s_demo.shape[0])
            epoch_loss += total_loss.item()
            epoch_rec_loss += rec_loss.item()
            epoch_vq_loss += (commitment_loss.item() + encoder_commitment.item())
            epoch_token_correct += int((pred_actions == a_demo).sum().item())
            epoch_token_total += int(a_demo.numel())
            epoch_exact_match += int(torch.all(pred_actions == a_demo, dim=1).sum().item())
            epoch_sample_total += batch_size
            epoch_code_counts += np.bincount(indices.detach().cpu().numpy(), minlength=config.num_archetypes)
            epoch_logit_abs_max = max(epoch_logit_abs_max, float(action_logits.abs().max().item()))
            epoch_z_e_norm_sum += float(torch.norm(z_e, dim=1).mean().item()) * batch_size
            epoch_quantization_mse_sum += float(torch.mean((z_e - z_q_detached) ** 2).item()) * batch_size
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_rec = epoch_rec_loss / max(num_batches, 1)
        avg_vq = epoch_vq_loss / max(num_batches, 1)
        code_usage_summary = summarize_code_usage(epoch_code_counts)
        token_accuracy = float(epoch_token_correct / max(epoch_token_total, 1))
        exact_match_rate = float(epoch_exact_match / max(epoch_sample_total, 1))
        avg_encoder_grad = float(epoch_encoder_grad / max(num_batches, 1))
        avg_codebook_grad = float(epoch_codebook_grad / max(num_batches, 1))
        avg_decoder_grad = float(epoch_decoder_grad / max(num_batches, 1))
        avg_z_e_norm = float(epoch_z_e_norm_sum / max(epoch_sample_total, 1))
        avg_quantization_mse = float(epoch_quantization_mse_sum / max(epoch_sample_total, 1))

        loss_history.append(avg_loss)
        rec_loss_history.append(avg_rec)
        vq_loss_history.append(avg_vq)
        token_accuracy_history.append(token_accuracy)
        exact_match_history.append(exact_match_rate)
        codebook_perplexity_history.append(code_usage_summary["codebook_perplexity"])
        used_code_count_history.append(code_usage_summary["used_code_count"])
        dominant_code_ratio_history.append(code_usage_summary["dominant_code_ratio"])
        encoder_grad_norm_history.append(avg_encoder_grad)
        codebook_grad_norm_history.append(avg_codebook_grad)
        decoder_grad_norm_history.append(avg_decoder_grad)
        logit_abs_max_history.append(epoch_logit_abs_max)
        z_e_norm_history.append(avg_z_e_norm)
        quantization_mse_history.append(avg_quantization_mse)

        # 每 10 个 epoch 或首尾 epoch 输出日志
        if epoch == 1 or epoch % 10 == 0 or epoch == config.phase1_epochs:
            logger.info(
                "Epoch %3d/%d — total_loss=%.4f, rec_loss=%.4f, vq_loss=%.4f, token_acc=%.4f, exact_match=%.4f, perplexity=%.4f, used_codes=%d",
                epoch,
                config.phase1_epochs,
                avg_loss,
                avg_rec,
                avg_vq,
                token_accuracy,
                exact_match_rate,
                code_usage_summary["codebook_perplexity"],
                code_usage_summary["used_code_count"],
            )

        # NaN 检测
        if np.isnan(avg_loss):
            logger.error("训练 loss 发散 (NaN)，在 epoch %d 终止训练", epoch)
            break

    training_monitor = {
        "loss_history": loss_history,
        "rec_loss_history": rec_loss_history,
        "vq_loss_history": vq_loss_history,
        "token_accuracy_history": token_accuracy_history,
        "exact_match_history": exact_match_history,
        "codebook_perplexity_history": codebook_perplexity_history,
        "used_code_count_history": used_code_count_history,
        "dominant_code_ratio_history": dominant_code_ratio_history,
        "encoder_grad_norm_history": encoder_grad_norm_history,
        "codebook_grad_norm_history": codebook_grad_norm_history,
        "decoder_grad_norm_history": decoder_grad_norm_history,
        "logit_abs_max_history": logit_abs_max_history,
        "z_e_norm_history": z_e_norm_history,
        "quantization_mse_history": quantization_mse_history,
    }

    # ----------------------------------------------------------------
    # Step 5: 保存模型到 result/phase1_archetype_discovery/
    # ----------------------------------------------------------------
    save_dir = os.path.join(config.result_dir, pair, "phase1_archetype_discovery")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{pair}_vq_model.pt")
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "codebook": codebook.state_dict(),
            "decoder": decoder.state_dict(),
            "loss_history": loss_history,
            "training_monitor": training_monitor,
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
                "num_trajectories": config.num_trajectories,
                "phase1_sampling_seed": config.phase1_sampling_seed,
                "discount_factor": config.discount_factor,
                "commission_rate": config.commission_rate,
                "max_positions": config.max_positions,
            },
        },
        save_path,
    )
    logger.info("模型已保存到 %s", save_path)

    # ----------------------------------------------------------------
    # Step 6: 执行 Phase I 验证并保存报告
    # ----------------------------------------------------------------
    report_path = os.path.join(save_dir, "phase1_validation_report.json")
    validation_report = validate_phase1_artifacts(
        config=config,
        pair=pair,
        trajectory_path=traj_path,
        model_path=save_path,
        report_path=report_path,
        env=env,
        device=device,
        dp_check_limit=256,
    )

    # ----------------------------------------------------------------
    # Step 7: 输出训练日志摘要
    # ----------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("Phase I 训练完成: pair=%s", pair)
    logger.info("最终 loss: %.4f", loss_history[-1] if loss_history else float("nan"))
    logger.info(
        "最低 loss: %.4f (epoch %d)",
        min(loss_history) if loss_history else float("nan"),
        (loss_history.index(min(loss_history)) + 1) if loss_history else 0,
    )
    logger.info("轨迹缓存路径: %s", traj_path)
    logger.info("模型保存路径: %s", save_path)
    logger.info("验证报告路径: %s", report_path)
    logger.info("验证是否通过: %s", validation_report["status"]["overall_passed"])
    if validation_report["status"]["hard_failures"]:
        logger.error("Phase I 验证硬失败: %s", validation_report["status"]["hard_failures"])
    if validation_report["status"]["soft_warnings"]:
        logger.warning("Phase I 验证软告警: %s", validation_report["status"]["soft_warnings"])
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
