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
import shutil
from datetime import datetime
from typing import Any, Dict, List, Tuple

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

PAPER_PHASE1_REFERENCE_TRAIN_ROWS = 1_400_000
PAPER_PHASE1_SPEC = {
    "state_dim": 45,
    "action_dim": 3,
    "horizon": 72,
    "commission_rate": 0.0002,
    "lstm_hidden_dim": 128,
    "latent_dim": 16,
    "num_archetypes": 10,
    "vq_beta0": 0.25,
    "num_trajectories": 30000,
    "phase1_epochs": 100,
    "discount_factor": 0.99,
    "max_positions": {"BTC": 8, "ETH": 100, "DOT": 2500, "BNB": 200},
}

# -----------------------------------------------------------------------------
# Paper-consistent debug branch v4 超参数
# -----------------------------------------------------------------------------
# 说明:
# 1. 仍不改变论文主流程、模型接口、最近邻量化公式或联合损失。
# 2. 在 v3 的基础上，进一步把主攻点放到 encoder 表征可分性：
#    - encoder 改为时间维 mean/max pooling summary；
#    - 正式 VQ 训练前先做短暂的连续 latent 重建预训练；
#    - 最终仍回到论文原来的最近邻量化与联合损失。
PAPER_CONSISTENT_DEBUG = {
    "decoder_lr_multiplier": 0.35,
    "grad_clip_max_norm": 1.0,
    "latent_init_max_samples": 8192,
    "latent_init_batch_size": 512,
    "reward_norm_eps": 1e-6,
    "continuous_pretrain_epochs": 8,
    "continuous_pretrain_decoder_lr_multiplier": 0.50,
    "vq_loss_warmup_epochs": 10,
    "vq_loss_warmup_start": 0.0,
    "vq_loss_warmup_end": 1.0,
}


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


def compute_linear_warmup_weight(
    current_epoch: int,
    warmup_epochs: int,
    start_weight: float = 0.0,
    end_weight: float = 1.0,
) -> float:
    """计算线性 warmup 权重。

    该 warmup 只作用于 VQ 两项在训练前期的优化强度，最终仍回到论文原式
    `L = L_rec + ||sg[z_e]-z_q||^2 + beta0 * ||z_e-sg[z_q]||^2`。

    Args:
        current_epoch: 当前 epoch，从 1 开始。
        warmup_epochs: warmup 持续 epoch 数；<=0 时直接返回 end_weight。
        start_weight: 起始权重。
        end_weight: 结束权重。
    """
    if warmup_epochs <= 0:
        return float(end_weight)
    progress = min(max(float(current_epoch), 1.0), float(warmup_epochs)) / float(warmup_epochs)
    return float(start_weight + (end_weight - start_weight) * progress)


def estimate_reward_normalization_from_dataloader(
    dataloader: DataLoader,
) -> Dict[str, float]:
    """估计训练集 reward 的标准化统计量。

    该统计量只用于实现层的输入尺度稳定，不改变论文中的 reward 定义或
    encoder 输入接口 `(s_demo, a_demo, r_demo)`。

    Args:
        dataloader: 不打乱顺序的轨迹数据加载器。

    Returns:
        包含 reward_mean / reward_std / reward_min / reward_max / reward_count 的字典。
    """
    total_count = 0
    total_sum = 0.0
    total_sq_sum = 0.0
    reward_min = float("inf")
    reward_max = float("-inf")

    for _, _, r_demo in dataloader:
        reward_array = r_demo.detach().cpu().numpy().astype(np.float64, copy=False)
        if reward_array.size <= 0:
            continue
        total_count += int(reward_array.size)
        total_sum += float(np.sum(reward_array))
        total_sq_sum += float(np.sum(np.square(reward_array)))
        reward_min = min(reward_min, float(np.min(reward_array)))
        reward_max = max(reward_max, float(np.max(reward_array)))

    if total_count <= 0:
        raise ValueError("无法从 dataloader 中估计 reward 标准化统计量：reward 总数为 0")

    reward_mean = total_sum / float(total_count)
    reward_var = max(total_sq_sum / float(total_count) - reward_mean * reward_mean, 0.0)
    reward_std = max(float(np.sqrt(reward_var)), float(PAPER_CONSISTENT_DEBUG["reward_norm_eps"]))

    return {
        "reward_mean": float(reward_mean),
        "reward_std": float(reward_std),
        "reward_min": float(reward_min),
        "reward_max": float(reward_max),
        "reward_count": int(total_count),
    }



def train_continuous_latent_pretrain(
    encoder: VQEncoder,
    decoder: VQDecoder,
    dataloader: DataLoader,
    device: torch.device,
    action_dim: int,
    encoder_lr: float,
    decoder_lr_multiplier: float,
    grad_clip_max_norm: float,
    num_epochs: int,
) -> Dict[str, Any]:
    """先用连续 z_e 做短暂重建预训练，再进入 VQ 训练。

    该阶段不改变最终论文模型；它只是在正式最近邻量化前，
    先让 encoder/decoder 学出非平凡的连续 latent 几何结构，
    以降低“量化前即单簇塌缩”的风险。

    Args:
        encoder: Phase I 编码器。
        decoder: Phase I 解码器。
        dataloader: 轨迹数据加载器。
        device: 运行设备。
        action_dim: 动作空间大小。
        encoder_lr: 预训练阶段 encoder 学习率。
        decoder_lr_multiplier: decoder 学习率缩放系数。
        grad_clip_max_norm: 梯度裁剪阈值。
        num_epochs: 连续 latent 预训练轮数。

    Returns:
        记录预训练损失与精度历史的字典。
    """
    summary: Dict[str, Any] = {
        "enabled": bool(num_epochs > 0),
        "num_epochs": int(max(num_epochs, 0)),
        "loss_history": [],
        "token_accuracy_history": [],
        "exact_match_history": [],
        "z_e_feature_std_history": [],
        "encoder_lr": float(encoder_lr),
        "decoder_lr": float(encoder_lr) * float(decoder_lr_multiplier),
    }
    if num_epochs <= 0:
        return summary

    optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters(), "lr": float(encoder_lr)},
            {"params": decoder.parameters(), "lr": float(encoder_lr) * float(decoder_lr_multiplier)},
        ]
    )
    ce_loss_fn = nn.CrossEntropyLoss()

    logger.info(
        "开始连续 latent 预训练: epochs=%d, encoder_lr=%.2e, decoder_lr=%.2e",
        num_epochs,
        float(encoder_lr),
        float(encoder_lr) * float(decoder_lr_multiplier),
    )

    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()

        epoch_loss = 0.0
        epoch_token_correct = 0
        epoch_token_total = 0
        epoch_exact_match = 0
        epoch_sample_total = 0
        epoch_z_e_feature_std_sum = 0.0
        num_batches = 0

        for s_demo, a_demo, r_demo in dataloader:
            s_demo = s_demo.to(device)
            a_demo = a_demo.to(device)
            r_demo = r_demo.to(device)

            z_e = encoder(s_demo, a_demo, r_demo)
            action_logits = decoder(s_demo, z_e)
            pred_actions = torch.argmax(action_logits, dim=-1)

            rec_loss = ce_loss_fn(action_logits.reshape(-1, action_dim), a_demo.reshape(-1))

            optimizer.zero_grad()
            rec_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=grad_clip_max_norm)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()

            batch_size = int(s_demo.shape[0])
            epoch_loss += float(rec_loss.item())
            epoch_token_correct += int((pred_actions == a_demo).sum().item())
            epoch_token_total += int(a_demo.numel())
            epoch_exact_match += int(torch.all(pred_actions == a_demo, dim=1).sum().item())
            epoch_sample_total += batch_size
            epoch_z_e_feature_std_sum += float(torch.std(z_e.detach(), dim=0).mean().item()) * batch_size
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        token_accuracy = float(epoch_token_correct / max(epoch_token_total, 1))
        exact_match_rate = float(epoch_exact_match / max(epoch_sample_total, 1))
        avg_z_e_feature_std = float(epoch_z_e_feature_std_sum / max(epoch_sample_total, 1))

        summary["loss_history"].append(avg_loss)
        summary["token_accuracy_history"].append(token_accuracy)
        summary["exact_match_history"].append(exact_match_rate)
        summary["z_e_feature_std_history"].append(avg_z_e_feature_std)

        if epoch == 1 or epoch == num_epochs or epoch % 5 == 0:
            logger.info(
                "预训练 Epoch %2d/%d — rec_loss=%.4f, token_acc=%.4f, exact_match=%.4f, z_e_feature_std=%.6f",
                epoch,
                num_epochs,
                avg_loss,
                token_accuracy,
                exact_match_rate,
                avg_z_e_feature_std,
            )

    return summary


def collect_encoder_latents_for_init(
    encoder: VQEncoder,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 4096,
) -> np.ndarray:
    """收集 encoder 产生的初始 z_e，用于 data-driven codebook 初始化。

    该方法严格沿用论文中的 encoder 接口 `q_{theta_e}(z_e | s_demo, a_demo, r_demo)`，
    只把随机初始化后的 encoder 输出作为码本初始位置的参考，不改变量化或损失定义。

    Args:
        encoder: Phase I 编码器。
        dataloader: 轨迹数据加载器。
        device: 运行设备。
        max_samples: 最多收集多少条 latent。

    Returns:
        shape = (num_collected, latent_dim) 的 numpy 数组。
    """
    if max_samples <= 0:
        raise ValueError(f"max_samples 必须为正整数，实际为 {max_samples}")

    was_training = encoder.training
    encoder.eval()
    collected: List[np.ndarray] = []
    num_collected = 0

    with torch.no_grad():
        for s_demo, a_demo, r_demo in dataloader:
            s_demo = s_demo.to(device)
            a_demo = a_demo.to(device)
            r_demo = r_demo.to(device)
            z_e = encoder(s_demo, a_demo, r_demo)
            z_e_np = z_e.detach().cpu().numpy()
            collected.append(z_e_np)
            num_collected += int(z_e_np.shape[0])
            if num_collected >= max_samples:
                break

    if was_training:
        encoder.train()

    if not collected:
        raise ValueError("未能从 dataloader 收集到任何 encoder latent，无法进行 codebook 初始化")

    latents = np.concatenate(collected, axis=0)[:max_samples]
    return latents.astype(np.float32, copy=False)


def build_data_driven_codebook_init(
    latent_vectors: np.ndarray,
    num_codes: int,
    seed: int,
) -> np.ndarray:
    """基于 encoder latent 构造码本初始向量。

    算法采用 farthest-point sampling：
    1. 随机选取第一个中心。
    2. 之后每次选择与当前中心集合最远的样本。

    该方法只决定 `e_0, ..., e_{K-1}` 的初始位置，不改变论文中的
    最近邻量化公式 `k = argmin_j ||z_e - e_j||^2`。

    Args:
        latent_vectors: shape = (N, latent_dim) 的候选 z_e。
        num_codes: 码本大小 K。
        seed: 随机种子，用于首个中心抽样。

    Returns:
        shape = (num_codes, latent_dim) 的初始化矩阵。
    """
    if latent_vectors.ndim != 2:
        raise ValueError(f"latent_vectors 应为 2D 数组，实际 ndim={latent_vectors.ndim}")
    if latent_vectors.shape[0] <= 0:
        raise ValueError("latent_vectors 为空，无法构造 codebook 初始化")
    if num_codes <= 0:
        raise ValueError(f"num_codes 必须为正整数，实际为 {num_codes}")

    candidates = np.asarray(latent_vectors, dtype=np.float32)
    if candidates.shape[0] < num_codes:
        repeats = int(np.ceil(float(num_codes) / float(candidates.shape[0])))
        candidates = np.tile(candidates, (repeats, 1))

    rng = np.random.default_rng(seed)
    first_idx = int(rng.integers(0, candidates.shape[0]))
    selected_indices = [first_idx]

    min_distance_sq = np.sum((candidates - candidates[first_idx]) ** 2, axis=1)
    for _ in range(1, num_codes):
        next_idx = int(np.argmax(min_distance_sq))
        selected_indices.append(next_idx)
        next_distance_sq = np.sum((candidates - candidates[next_idx]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, next_distance_sq)

    return candidates[selected_indices].astype(np.float32, copy=True)


def summarize_codebook_init(init_codes: np.ndarray) -> Dict[str, float]:
    """汇总 data-driven codebook 初始化的几何统计。"""
    if init_codes.ndim != 2:
        raise ValueError(f"init_codes 应为 2D 数组，实际 ndim={init_codes.ndim}")

    code_norms = np.linalg.norm(init_codes, axis=1)
    if init_codes.shape[0] >= 2:
        pairwise_l2 = []
        for i in range(init_codes.shape[0]):
            for j in range(i + 1, init_codes.shape[0]):
                pairwise_l2.append(float(np.linalg.norm(init_codes[i] - init_codes[j])))
        pairwise_l2_array = np.asarray(pairwise_l2, dtype=np.float64)
    else:
        pairwise_l2_array = np.empty(0, dtype=np.float64)

    return {
        "init_code_norm_mean": float(np.mean(code_norms)) if code_norms.size else 0.0,
        "init_code_norm_std": float(np.std(code_norms)) if code_norms.size else 0.0,
        "pairwise_l2_mean": float(np.mean(pairwise_l2_array)) if pairwise_l2_array.size else 0.0,
        "pairwise_l2_min": float(np.min(pairwise_l2_array)) if pairwise_l2_array.size else 0.0,
        "pairwise_l2_max": float(np.max(pairwise_l2_array)) if pairwise_l2_array.size else 0.0,
    }



def assert_paper_phase1_settings(config: Any, pair: str) -> None:
    """强制检查当前配置是否严格等于论文 Phase I 主实验设置。

    该守卫不改变论文算法，只负责阻止“看起来在复现论文、实际上配置已偏离”的运行。
    若用户希望做非论文设置实验，应显式修改该守卫或另建 debug 分支，而不是在主线脚本中静默放宽。

    Args:
        config: 解析后的运行配置。
        pair: 当前交易对名称。

    Raises:
        ValueError: 当任一关键配置与论文主实验不一致时抛出。
    """
    mismatches: List[str] = []

    def _check_exact(name: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            mismatches.append(f"{name}: actual={actual}, expected={expected}")

    def _check_float(name: str, actual: float, expected: float, atol: float = 1e-12) -> None:
        if not np.isclose(actual, expected, atol=atol, rtol=0.0):
            mismatches.append(f"{name}: actual={actual}, expected={expected}")

    _check_exact("state_dim", config.state_dim, PAPER_PHASE1_SPEC["state_dim"])
    _check_exact("action_dim", config.action_dim, PAPER_PHASE1_SPEC["action_dim"])
    _check_exact("horizon", config.horizon, PAPER_PHASE1_SPEC["horizon"])
    _check_float("commission_rate", config.commission_rate, PAPER_PHASE1_SPEC["commission_rate"])
    _check_exact("lstm_hidden_dim", config.lstm_hidden_dim, PAPER_PHASE1_SPEC["lstm_hidden_dim"])
    _check_exact("latent_dim", config.latent_dim, PAPER_PHASE1_SPEC["latent_dim"])
    _check_exact("num_archetypes", config.num_archetypes, PAPER_PHASE1_SPEC["num_archetypes"])
    _check_float("vq_beta0", config.vq_beta0, PAPER_PHASE1_SPEC["vq_beta0"])
    _check_exact("num_trajectories", config.num_trajectories, PAPER_PHASE1_SPEC["num_trajectories"])
    _check_exact("phase1_epochs", config.phase1_epochs, PAPER_PHASE1_SPEC["phase1_epochs"])
    _check_float("discount_factor", config.discount_factor, PAPER_PHASE1_SPEC["discount_factor"])

    expected_m = PAPER_PHASE1_SPEC["max_positions"].get(pair)
    actual_m = config.max_positions.get(pair)
    _check_exact(f"max_positions[{pair}]", actual_m, expected_m)

    if mismatches:
        joined = "\n  - ".join(mismatches)
        raise ValueError(
            "当前运行配置不是严格论文 Phase I 主实验设置，已停止训练。\n"
            f"  - {joined}"
        )



def log_training_data_scale(train_rows: int) -> None:
    """记录当前训练数据规模与论文规模的差异。

    该日志只用于解释“当前是 reduced-data reproduction”，不会阻止训练。
    论文约使用 140 万行训练数据；用户当前约 52 万行，因此需要把数据规模差异与算法差异分开。
    """
    ratio = float(train_rows) / float(PAPER_PHASE1_REFERENCE_TRAIN_ROWS)
    logger.warning(
        "当前训练集行数=%d，论文约使用=%d 行；当前约为论文数据规模的 %.2f%%。"
        "这仍属于严格论文算法/公式下的 reduced-data reproduction，而非同数据规模复现。",
        train_rows,
        PAPER_PHASE1_REFERENCE_TRAIN_ROWS,
        ratio * 100.0,
    )



def expected_num_available_starts(total_rows: int, horizon: int) -> int:
    """计算滑窗采样协议下全部合法起点数量。"""
    return max(total_rows - horizon + 1, 0)



def inspect_trajectory_cache(
    traj_path: str,
    config: Any,
    pair: str,
    train_rows: int,
) -> Tuple[bool, List[str]]:
    """检查现有轨迹缓存是否与当前严格论文设置兼容。

    检查项覆盖 pair / horizon / gamma / num_trajectories / sampling_seed /
    训练集长度 / state_dim / commission_rate / max_position / 算法变体标记，
    从而避免误复用旧缓存导致“日志看起来一样、实际并未重跑”。

    Args:
        traj_path: 轨迹缓存路径。
        config: 当前运行配置。
        pair: 当前交易对。
        train_rows: 当前训练集行数。

    Returns:
        (is_compatible, reasons)
    """
    if not os.path.exists(traj_path):
        return False, ["trajectory cache 文件不存在"]

    reasons: List[str] = []
    expected_starts = expected_num_available_starts(train_rows, config.horizon)
    expected_values = {
        "pair": pair,
        "horizon": int(config.horizon),
        "gamma": float(config.discount_factor),
        "num_sampled_trajectories": int(config.num_trajectories),
        "sampling_seed": int(config.phase1_sampling_seed),
        "num_available_starts": int(expected_starts),
        "training_rows": int(train_rows),
        "state_dim": int(config.state_dim),
        "commission_rate": float(config.commission_rate),
        "max_position": int(config.max_positions[pair]),
        "algorithm_variant": "paper_single_change",
    }

    with np.load(traj_path, allow_pickle=False) as data:
        required_keys = set(expected_values.keys()) | {"states", "actions", "rewards"}
        missing_keys = sorted(required_keys - set(data.files))
        if missing_keys:
            reasons.append(f"cache 缺少关键元数据: {missing_keys}")
            return False, reasons

        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        if states.ndim != 3 or states.shape[1] != config.horizon or states.shape[2] != config.state_dim:
            reasons.append(
                f"states shape 不匹配: actual={states.shape}, expected=(*, {config.horizon}, {config.state_dim})"
            )
        if actions.ndim != 2 or actions.shape[0] != states.shape[0] or actions.shape[1] != config.horizon:
            reasons.append(
                f"actions shape 不匹配: actual={actions.shape}, expected=({states.shape[0]}, {config.horizon})"
            )
        if rewards.ndim != 2 or rewards.shape[0] != states.shape[0] or rewards.shape[1] != config.horizon:
            reasons.append(
                f"rewards shape 不匹配: actual={rewards.shape}, expected=({states.shape[0]}, {config.horizon})"
            )

        for key, expected in expected_values.items():
            raw_value = data[key]
            if isinstance(raw_value, np.ndarray) and raw_value.shape == ():
                actual = raw_value.item()
            elif isinstance(raw_value, np.ndarray) and raw_value.size == 1:
                actual = raw_value.reshape(()).item()
            else:
                actual = raw_value
            if isinstance(actual, bytes):
                actual = actual.decode("utf-8")
            if isinstance(expected, float):
                if not np.isclose(float(actual), expected, atol=1e-12, rtol=0.0):
                    reasons.append(f"{key} 不匹配: actual={actual}, expected={expected}")
            else:
                if actual != expected:
                    reasons.append(f"{key} 不匹配: actual={actual}, expected={expected}")

    return len(reasons) == 0, reasons



def backup_incompatible_cache(traj_path: str, reasons: List[str]) -> str:
    """备份不兼容的旧轨迹缓存，避免被当前严格论文运行误复用。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = traj_path.replace(".npz", f".incompatible_{timestamp}.npz")
    shutil.move(traj_path, backup_path)
    logger.warning(
        "检测到现有 trajectory cache 与当前严格论文设置不兼容，已备份到 %s。原因: %s",
        backup_path,
        reasons,
    )
    return backup_path



def main() -> None:
    # ----------------------------------------------------------------
    # Step 0: 解析配置
    # ----------------------------------------------------------------
    config = parse_args()
    pair = config.pairs[0]  # 单交易对训练
    assert_paper_phase1_settings(config, pair)
    set_reproducibility_seed(config.phase1_sampling_seed)

    logger.info("Phase I 训练开始: pair=%s", pair)
    logger.info(
        "严格论文主线配置已通过守卫检查: epochs=%d, batch_size=%d, lr=%.1e, latent_dim=%d, "
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
    train_rows = int(train_states.shape[0])

    logger.info(
        "训练集: states shape=%s, prices shape=%s",
        train_states.shape,
        prices.shape,
    )
    log_training_data_scale(train_rows)

    available_starts = expected_num_available_starts(train_rows, config.horizon)
    if available_starts < config.num_trajectories:
        raise ValueError(
            "当前训练集不足以在严格论文滑窗协议下无放回采样 30k trajectories。"
            f" available_starts={available_starts}, required={config.num_trajectories}"
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
        "num_horizons=%d, horizon=%d, max_position=%d, commission_rate=%.6f, available_starts=%d",
        env.num_horizons,
        config.horizon,
        env.m,
        env.commission_rate,
        available_starts,
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

    need_generate_trajectories = True
    if os.path.exists(traj_path):
        cache_ok, cache_reasons = inspect_trajectory_cache(
            traj_path=traj_path,
            config=config,
            pair=pair,
            train_rows=train_rows,
        )
        if cache_ok:
            logger.info("发现与当前严格论文设置兼容的轨迹缓存，直接加载: %s", traj_path)
            dataset = TrajectoryDataset.from_npz(traj_path)
            need_generate_trajectories = False
        else:
            backup_incompatible_cache(traj_path, cache_reasons)

    if need_generate_trajectories:
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
    init_dataloader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, PAPER_CONSISTENT_DEBUG["latent_init_batch_size"]),
        shuffle=False,
        drop_last=False,
    )

    reward_norm_summary = estimate_reward_normalization_from_dataloader(init_dataloader)
    logger.info(
        "训练 reward 标准化统计: mean=%.6f, std=%.6f, min=%.6f, max=%.6f, count=%d",
        reward_norm_summary["reward_mean"],
        reward_norm_summary["reward_std"],
        reward_norm_summary["reward_min"],
        reward_norm_summary["reward_max"],
        reward_norm_summary["reward_count"],
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

    encoder.set_reward_normalization(
        reward_mean=reward_norm_summary["reward_mean"],
        reward_std=reward_norm_summary["reward_std"],
        eps=PAPER_CONSISTENT_DEBUG["reward_norm_eps"],
    )

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
        "模型初始化完成: Encoder params=%d, Codebook params=%d, Decoder params=%d, encoder_reward_std=%.6f",
        sum(p.numel() for p in encoder.parameters()),
        sum(p.numel() for p in codebook.parameters()),
        sum(p.numel() for p in decoder.parameters()),
        encoder.get_reward_normalization()["reward_std"],
    )

    pretrain_summary = train_continuous_latent_pretrain(
        encoder=encoder,
        decoder=decoder,
        dataloader=dataloader,
        device=device,
        action_dim=config.action_dim,
        encoder_lr=float(config.learning_rate),
        decoder_lr_multiplier=float(PAPER_CONSISTENT_DEBUG["continuous_pretrain_decoder_lr_multiplier"]),
        grad_clip_max_norm=float(PAPER_CONSISTENT_DEBUG["grad_clip_max_norm"]),
        num_epochs=int(PAPER_CONSISTENT_DEBUG["continuous_pretrain_epochs"]),
    )

    # Paper-consistent debug: 预训练后，用 encoder 产生的 z_e 初始化 codebook。
    init_latents = collect_encoder_latents_for_init(
        encoder=encoder,
        dataloader=init_dataloader,
        device=device,
        max_samples=PAPER_CONSISTENT_DEBUG["latent_init_max_samples"],
    )
    init_codes = build_data_driven_codebook_init(
        latent_vectors=init_latents,
        num_codes=config.num_archetypes,
        seed=config.phase1_sampling_seed,
    )
    codebook.set_codebook_vectors(torch.from_numpy(init_codes).to(device=device))
    codebook_init_summary = summarize_codebook_init(init_codes)
    logger.info(
        "完成预训练后 data-driven codebook 初始化: max_samples=%d, init_code_norm_mean=%.4f, pairwise_l2_min=%.4f, pairwise_l2_mean=%.4f",
        min(PAPER_CONSISTENT_DEBUG["latent_init_max_samples"], int(init_latents.shape[0])),
        codebook_init_summary["init_code_norm_mean"],
        codebook_init_summary["pairwise_l2_min"],
        codebook_init_summary["pairwise_l2_mean"],
    )

    # 优化器：联合训练 encoder + codebook + decoder。
    # 这里仅调整参数组学习率，不改变论文的损失项与前向/量化流程。
    encoder_lr = float(config.learning_rate)
    codebook_lr = float(config.learning_rate)
    decoder_lr = float(config.learning_rate) * float(PAPER_CONSISTENT_DEBUG["decoder_lr_multiplier"])
    optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters(), "lr": encoder_lr},
            {"params": codebook.parameters(), "lr": codebook_lr},
            {"params": decoder.parameters(), "lr": decoder_lr},
        ]
    )
    logger.info(
        "优化器参数组: encoder_lr=%.2e, codebook_lr=%.2e, decoder_lr=%.2e, grad_clip_max_norm=%.2f",
        encoder_lr,
        codebook_lr,
        decoder_lr,
        PAPER_CONSISTENT_DEBUG["grad_clip_max_norm"],
    )

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
    z_e_feature_std_history = []
    vq_loss_weight_history = []

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
        epoch_z_e_feature_std_sum = 0.0
        num_batches = 0

        vq_loss_weight = compute_linear_warmup_weight(
            current_epoch=epoch,
            warmup_epochs=int(PAPER_CONSISTENT_DEBUG["vq_loss_warmup_epochs"]),
            start_weight=float(PAPER_CONSISTENT_DEBUG["vq_loss_warmup_start"]),
            end_weight=float(PAPER_CONSISTENT_DEBUG["vq_loss_warmup_end"]),
        )

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
            weighted_codebook_loss = vq_loss_weight * commitment_loss
            weighted_encoder_commitment = vq_loss_weight * encoder_commitment

            # 总损失: 训练前期对 VQ 两项做线性 warmup，最终恢复到论文原式。
            # 当 warmup 权重达到 1 后，
            # L = L_rec + ||sg[z_e] - z_q||² + beta0 * ||z_e - sg[z_q]||²
            total_loss = rec_loss + weighted_codebook_loss + weighted_encoder_commitment

            optimizer.zero_grad()
            total_loss.backward()

            epoch_encoder_grad += compute_grad_norm(encoder.parameters())
            epoch_codebook_grad += compute_grad_norm(codebook.parameters())
            epoch_decoder_grad += compute_grad_norm(decoder.parameters())

            torch.nn.utils.clip_grad_norm_(
                encoder.parameters(),
                max_norm=PAPER_CONSISTENT_DEBUG["grad_clip_max_norm"],
            )
            torch.nn.utils.clip_grad_norm_(
                codebook.parameters(),
                max_norm=PAPER_CONSISTENT_DEBUG["grad_clip_max_norm"],
            )
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                max_norm=PAPER_CONSISTENT_DEBUG["grad_clip_max_norm"],
            )

            optimizer.step()

            batch_size = int(s_demo.shape[0])
            epoch_loss += total_loss.item()
            epoch_rec_loss += rec_loss.item()
            epoch_vq_loss += (weighted_codebook_loss.item() + weighted_encoder_commitment.item())
            epoch_token_correct += int((pred_actions == a_demo).sum().item())
            epoch_token_total += int(a_demo.numel())
            epoch_exact_match += int(torch.all(pred_actions == a_demo, dim=1).sum().item())
            epoch_sample_total += batch_size
            epoch_code_counts += np.bincount(indices.detach().cpu().numpy(), minlength=config.num_archetypes)
            epoch_logit_abs_max = max(epoch_logit_abs_max, float(action_logits.abs().max().item()))
            epoch_z_e_norm_sum += float(torch.norm(z_e, dim=1).mean().item()) * batch_size
            epoch_quantization_mse_sum += float(torch.mean((z_e - z_q_detached) ** 2).item()) * batch_size
            epoch_z_e_feature_std_sum += float(torch.std(z_e.detach(), dim=0).mean().item()) * batch_size
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
        avg_z_e_feature_std = float(epoch_z_e_feature_std_sum / max(epoch_sample_total, 1))

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
        z_e_feature_std_history.append(avg_z_e_feature_std)
        vq_loss_weight_history.append(vq_loss_weight)

        # 每 10 个 epoch 或首尾 epoch 输出日志
        if epoch == 1 or epoch % 10 == 0 or epoch == config.phase1_epochs:
            logger.info(
                "Epoch %3d/%d — total_loss=%.4f, rec_loss=%.4f, vq_loss=%.4f, vq_w=%.3f, token_acc=%.4f, exact_match=%.4f, perplexity=%.4f, used_codes=%d",
                epoch,
                config.phase1_epochs,
                avg_loss,
                avg_rec,
                avg_vq,
                vq_loss_weight,
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
        "z_e_feature_std_history": z_e_feature_std_history,
        "vq_loss_weight_history": vq_loss_weight_history,
        "optimizer_group_learning_rates": {
            "encoder": encoder_lr,
            "codebook": codebook_lr,
            "decoder": decoder_lr,
        },
        "paper_consistent_debug": {
            "decoder_lr_multiplier": PAPER_CONSISTENT_DEBUG["decoder_lr_multiplier"],
            "grad_clip_max_norm": PAPER_CONSISTENT_DEBUG["grad_clip_max_norm"],
            "latent_init_max_samples": PAPER_CONSISTENT_DEBUG["latent_init_max_samples"],
            "latent_init_batch_size": PAPER_CONSISTENT_DEBUG["latent_init_batch_size"],
            "reward_norm_eps": PAPER_CONSISTENT_DEBUG["reward_norm_eps"],
            "continuous_pretrain_epochs": PAPER_CONSISTENT_DEBUG["continuous_pretrain_epochs"],
            "continuous_pretrain_decoder_lr_multiplier": PAPER_CONSISTENT_DEBUG["continuous_pretrain_decoder_lr_multiplier"],
            "vq_loss_warmup_epochs": PAPER_CONSISTENT_DEBUG["vq_loss_warmup_epochs"],
            "vq_loss_warmup_start": PAPER_CONSISTENT_DEBUG["vq_loss_warmup_start"],
            "vq_loss_warmup_end": PAPER_CONSISTENT_DEBUG["vq_loss_warmup_end"],
            "decoder_parameterization": "film_conditioned_mlp_decoder",
            "encoder_reward_normalization": "global_train_reward_standardization",
            "encoder_output_parameterization": "mean_max_temporal_pooling_projection_head",
            "vq_onset_strategy": "continuous_latent_pretraining_then_vq_finetuning",
        },
        "continuous_pretrain_summary": pretrain_summary,
        "codebook_init_summary": codebook_init_summary,
        "reward_normalization_summary": reward_norm_summary,
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
                "paper_phase1_reference_train_rows": PAPER_PHASE1_REFERENCE_TRAIN_ROWS,
                "current_train_rows": train_rows,
                "paper_consistent_debug": PAPER_CONSISTENT_DEBUG,
                "reward_normalization_summary": reward_norm_summary,
                "decoder_parameterization": "film_conditioned_mlp_decoder",
                "encoder_output_parameterization": "mean_max_temporal_pooling_projection_head",
                "vq_onset_strategy": "continuous_latent_pretraining_then_vq_finetuning",
                "continuous_pretrain_summary": pretrain_summary,
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
