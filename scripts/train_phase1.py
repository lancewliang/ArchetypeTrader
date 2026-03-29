#!/usr/bin/env python
"""Phase I 训练脚本 — 原型发现（debug branch）

# 需求: 7.1, 4.6, 4.7, 4.8, 7.5, 7.6, 7.7
#
# 流程:
# 1. 加载特征数据，初始化 TradingEnv
# 2. 调用 DPPlanner 生成 30k 示范轨迹并保存
# 3. 初始化 VQ Encoder、Codebook、Decoder
# 4. 执行 debug branch 预处理/稳定化：
#    4.1 reward normalization ablation（仅作用于 encoder 输入）
#    4.2 warm-start codebook 初始化
# 5. 训练 100 epochs
#    损失函数 L = L_rec + ||sg[z_e] - z_q||² + 0.25 × ||z_e - sg[z_q]||²
# 6. 执行 dead-code reinit（仅在长期未使用时触发，不改变训练目标）
# 7. 保存模型到 result/phase1_archetype_discovery/
# 8. 执行 Phase I 验证并保存 phase1_validation_report.json
#
# 用法:
#   python scripts/train_phase1.py --pair BTC
#   python scripts/train_phase1.py --pair ETH --phase1-reward-norm-mode zscore
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


def set_reproducibility_seed(seed: int) -> None:
    """设置 Phase I 复现实验所需的随机种子。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def compute_grad_norm(parameters) -> float:
    """计算参数梯度的全局 L2 范数。"""
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



def assert_paper_phase1_settings(config: Any, pair: str) -> None:
    """强制检查当前配置是否严格等于论文 Phase I 主实验设置。

    debug branch 允许在“训练流程/稳定化细节”上做实验，但基础主实验超参数
    仍然锁定为论文值，避免把超参偏差和训练稳定性问题混在一起。
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
    """记录当前训练数据规模与论文规模的差异。"""
    ratio = float(train_rows) / float(PAPER_PHASE1_REFERENCE_TRAIN_ROWS)
    logger.warning(
        "当前训练集行数=%d，论文约使用=%d 行；当前约为论文数据规模的 %.2f%%。"
        "这属于严格论文主实验超参数下的 debug/reduced-data reproduction。",
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
    """检查现有轨迹缓存是否与当前运行设置兼容。"""
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
    """备份不兼容的旧轨迹缓存，避免被当前运行误复用。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = traj_path.replace(".npz", f".incompatible_{timestamp}.npz")
    shutil.move(traj_path, backup_path)
    logger.warning(
        "检测到现有 trajectory cache 与当前设置不兼容，已备份到 %s。原因: %s",
        backup_path,
        reasons,
    )
    return backup_path



def build_reward_normalization_stats(dataset: TrajectoryDataset, config: Any) -> Dict[str, float]:
    """基于整个轨迹数据集构造 reward normalization 统计量。

    debug branch 中的 reward normalization 只作用于 encoder 输入，不改变 DP reward、
    decoder 重建目标，也不改变 VQ 损失定义。
    """
    rewards = dataset.rewards.detach().cpu().numpy().astype(np.float64)
    mean = float(np.mean(rewards)) if rewards.size else 0.0
    std = float(np.std(rewards)) if rewards.size else 1.0
    std = max(std, float(config.phase1_reward_norm_eps))
    return {
        "mode": config.phase1_reward_norm_mode,
        "mean": mean,
        "std": std,
        "eps": float(config.phase1_reward_norm_eps),
        "clip": float(config.phase1_reward_norm_clip),
        "raw_min": float(np.min(rewards)) if rewards.size else 0.0,
        "raw_max": float(np.max(rewards)) if rewards.size else 0.0,
    }



def apply_reward_normalization(r_demo: torch.Tensor, reward_norm_stats: Dict[str, float]) -> torch.Tensor:
    """对 encoder 输入 reward 进行标准化/裁剪。

    Args:
        r_demo: 原始 reward 序列。
        reward_norm_stats: 标准化配置与统计量。

    Returns:
        仅供 encoder 使用的 reward 输入。
    """
    mode = reward_norm_stats.get("mode", "none")
    if mode == "none":
        return r_demo

    normalized = (r_demo - float(reward_norm_stats["mean"])) / float(reward_norm_stats["std"])
    clip = float(reward_norm_stats.get("clip", 0.0))
    if clip > 0:
        normalized = torch.clamp(normalized, min=-clip, max=clip)
    return normalized



def collect_warm_start_latents(
    encoder: VQEncoder,
    dataloader: DataLoader,
    device: torch.device,
    reward_norm_stats: Dict[str, float],
    max_batches: int,
    max_samples: int,
) -> torch.Tensor:
    """收集 warm-start codebook 初始化所需的 latent 样本。"""
    encoder.eval()
    latents: List[torch.Tensor] = []
    collected = 0
    with torch.no_grad():
        for batch_idx, (s_demo, a_demo, r_demo) in enumerate(dataloader):
            if batch_idx >= max_batches or collected >= max_samples:
                break
            s_demo = s_demo.to(device)
            a_demo = a_demo.to(device)
            r_demo = apply_reward_normalization(r_demo.to(device), reward_norm_stats)
            z_e = encoder(s_demo, a_demo, r_demo)
            remaining = max_samples - collected
            latents.append(z_e[:remaining].detach().cpu())
            collected += min(int(z_e.shape[0]), remaining)
    if not latents:
        raise RuntimeError("warm-start latent 收集失败：没有可用 batch")
    return torch.cat(latents, dim=0)



def update_latent_reservoir(
    reservoir: List[torch.Tensor],
    z_e: torch.Tensor,
    max_size: int,
) -> None:
    """维护用于 dead-code reinit 的 latent reservoir。"""
    if max_size <= 0:
        return
    remaining = max_size - sum(int(chunk.shape[0]) for chunk in reservoir)
    if remaining <= 0:
        return
    reservoir.append(z_e[:remaining].detach().cpu())



def compute_latent_dependence_gap(
    decoder: VQDecoder,
    states: torch.Tensor,
    true_actions: torch.Tensor,
    z_q_st: torch.Tensor,
) -> Dict[str, float]:
    """监控 decoder 对 latent 的依赖程度。

    做法：
    1. 用真实 z_q 计算 token accuracy；
    2. 在 batch 内随机打乱 z_q，再计算一次 token accuracy；
    3. 两者差值越大，说明 decoder 越依赖 latent，而不只是依赖 states。
    """
    with torch.no_grad():
        real_logits = decoder(states, z_q_st)
        real_preds = torch.argmax(real_logits, dim=-1)
        real_acc = float((real_preds == true_actions).float().mean().item())

        if states.shape[0] <= 1:
            shuffled_acc = real_acc
        else:
            perm = torch.randperm(states.shape[0], device=states.device)
            shuffled_logits = decoder(states, z_q_st[perm])
            shuffled_preds = torch.argmax(shuffled_logits, dim=-1)
            shuffled_acc = float((shuffled_preds == true_actions).float().mean().item())

    return {
        "real_token_accuracy": real_acc,
        "shuffled_token_accuracy": shuffled_acc,
        "latent_dependence_gap": real_acc - shuffled_acc,
    }



def clear_optimizer_state_for_code_indices(
    optimizer: torch.optim.Optimizer,
    embedding_weight: torch.nn.Parameter,
    code_indices: List[int],
) -> None:
    """清理 Adam 中与指定 code 行对应的动量状态。"""
    if not code_indices:
        return
    state = optimizer.state.get(embedding_weight, None)
    if not state:
        return
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        tensor = state.get(key)
        if torch.is_tensor(tensor) and tensor.ndim >= 1 and tensor.shape[0] == embedding_weight.shape[0]:
            tensor[code_indices] = 0.0
    step = state.get("step")
    if torch.is_tensor(step) and step.ndim >= 1 and step.shape[0] == embedding_weight.shape[0]:
        step[code_indices] = 0



def maybe_reinitialize_dead_codes(
    codebook: VQCodebook,
    optimizer: torch.optim.Optimizer,
    latent_reservoir: List[torch.Tensor],
    dead_streaks: np.ndarray,
    epoch: int,
    config: Any,
) -> Dict[str, Any]:
    """在满足条件时执行 dead-code reinit。

    Args:
        codebook: 当前 codebook。
        optimizer: 联合优化器，用于同步清理被重置 code 的动量状态。
        latent_reservoir: 当前 epoch 收集到的 latent 样本。
        dead_streaks: 每个 code 连续未使用 epoch 计数。
        epoch: 当前 epoch。
        config: 运行配置。

    Returns:
        本轮 dead-code 处理摘要。
    """
    summary = {
        "epoch": int(epoch),
        "reinitialized": False,
        "code_indices": [],
        "num_source_vectors": 0,
    }
    if not config.phase1_dead_code_reinit:
        return summary
    if epoch < config.phase1_dead_code_min_epoch:
        return summary
    if not latent_reservoir:
        return summary

    candidate_codes = np.where(dead_streaks >= config.phase1_dead_code_patience)[0].astype(np.int64)
    if candidate_codes.size == 0:
        return summary

    selected_codes = candidate_codes[: int(config.phase1_dead_code_max_codes)]
    latent_pool = torch.cat(latent_reservoir, dim=0)
    reset_summary = codebook.reinitialize_codes_from_samples(
        code_indices=selected_codes.tolist(),
        latent_samples=latent_pool,
        seed=int(config.phase1_sampling_seed + epoch),
    )
    clear_optimizer_state_for_code_indices(optimizer, codebook.embeddings.weight, selected_codes.tolist())
    dead_streaks[selected_codes] = 0

    summary.update(
        {
            "reinitialized": True,
            "code_indices": [int(x) for x in selected_codes.tolist()],
            "num_source_vectors": int(reset_summary.get("num_source_vectors", 0)),
        }
    )
    return summary



def main() -> None:
    # ----------------------------------------------------------------
    # Step 0: 解析配置
    # ----------------------------------------------------------------
    config = parse_args()
    pair = config.pairs[0]  # 单交易对训练
    assert_paper_phase1_settings(config, pair)
    set_reproducibility_seed(config.phase1_sampling_seed)

    logger.info("Phase I debug branch 训练开始: pair=%s", pair)
    logger.info(
        "严格论文主实验配置已通过守卫检查: epochs=%d, batch_size=%d, lr=%.1e, latent_dim=%d, "
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
    logger.info(
        "debug branch 开关: reward_norm_mode=%s, warm_start=%s, dead_code_reinit=%s, latent_monitor=%s",
        config.phase1_reward_norm_mode,
        config.phase1_warm_start_codebook,
        config.phase1_dead_code_reinit,
        config.phase1_monitor_latent_dependence,
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
            logger.info("发现与当前设置兼容的轨迹缓存，直接加载: %s", traj_path)
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

    # debug branch: reward normalization 统计量
    reward_norm_stats = build_reward_normalization_stats(dataset, config)
    logger.info(
        "reward normalization: mode=%s, mean=%.6f, std=%.6f, clip=%.3f, raw_min=%.6f, raw_max=%.6f",
        reward_norm_stats["mode"],
        reward_norm_stats["mean"],
        reward_norm_stats["std"],
        reward_norm_stats["clip"],
        reward_norm_stats["raw_min"],
        reward_norm_stats["raw_max"],
    )

    # debug branch: codebook warm-start 初始化
    warm_start_summary: Dict[str, Any] = {
        "enabled": bool(config.phase1_warm_start_codebook),
        "applied": False,
        "num_source_vectors": 0,
    }
    if config.phase1_warm_start_codebook:
        warm_latents = collect_warm_start_latents(
            encoder=encoder,
            dataloader=dataloader,
            device=device,
            reward_norm_stats=reward_norm_stats,
            max_batches=int(config.phase1_warm_start_batches),
            max_samples=int(config.phase1_warm_start_max_samples),
        )
        warm_start_summary.update(
            codebook.initialize_from_samples(
                latent_samples=warm_latents,
                seed=int(config.phase1_sampling_seed),
            )
        )
        warm_start_summary["applied"] = True
        logger.info(
            "codebook warm-start 完成: source_vectors=%d, initialized_codes=%d",
            warm_start_summary.get("num_source_vectors", 0),
            warm_start_summary.get("num_initialized_codes", 0),
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
    latent_dependence_gap_history = []
    latent_real_acc_history = []
    latent_shuffled_acc_history = []
    dead_code_reinit_events: List[Dict[str, Any]] = []
    dead_code_streaks = np.zeros(config.num_archetypes, dtype=np.int64)

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
        epoch_latent_gap_values: List[float] = []
        epoch_latent_real_acc_values: List[float] = []
        epoch_latent_shuffle_acc_values: List[float] = []
        epoch_latent_reservoir: List[torch.Tensor] = []
        num_batches = 0

        for batch_idx, (s_demo, a_demo, r_demo) in enumerate(dataloader):
            s_demo = s_demo.to(device)   # (B, h, state_dim)
            a_demo = a_demo.to(device)   # (B, h)
            r_demo = r_demo.to(device)   # (B, h)
            r_demo_encoder = apply_reward_normalization(r_demo, reward_norm_stats)

            # Phase I, Step 1: Encode
            z_e = encoder(s_demo, a_demo, r_demo_encoder)  # (B, latent_dim)
            update_latent_reservoir(
                reservoir=epoch_latent_reservoir,
                z_e=z_e,
                max_size=int(config.phase1_dead_code_reservoir_size),
            )

            # Phase I, Step 2: Quantize
            z_q_st, indices, commitment_loss = codebook.quantize(z_e)
            # z_q_st: straight-through (B, latent_dim)
            # commitment_loss: ||sg[z_e] - z_q||²

            # Phase I, Step 3: Decode
            action_logits = decoder(s_demo, z_q_st)  # (B, h, action_dim)
            pred_actions = torch.argmax(action_logits, dim=-1)

            # L_rec: 交叉熵重建损失
            logits_flat = action_logits.reshape(-1, config.action_dim)
            targets_flat = a_demo.reshape(-1)
            rec_loss = ce_loss_fn(logits_flat, targets_flat)

            # β₀ × ||z_e - sg[z_q]||²
            z_q_detached = z_q_st.detach()  # sg[z_q] — stop gradient on z_q
            encoder_commitment = config.vq_beta0 * torch.mean(
                (z_e - z_q_detached) ** 2
            )

            # 总损失: L = L_rec + commitment_loss + β₀ × encoder_commitment
            total_loss = rec_loss + commitment_loss + encoder_commitment

            optimizer.zero_grad()
            total_loss.backward()

            epoch_encoder_grad += compute_grad_norm(encoder.parameters())
            epoch_codebook_grad += compute_grad_norm(codebook.parameters())
            epoch_decoder_grad += compute_grad_norm(decoder.parameters())

            optimizer.step()

            if config.phase1_monitor_latent_dependence and batch_idx < int(config.phase1_monitor_batches):
                gap_summary = compute_latent_dependence_gap(
                    decoder=decoder,
                    states=s_demo,
                    true_actions=a_demo,
                    z_q_st=z_q_st.detach(),
                )
                epoch_latent_gap_values.append(gap_summary["latent_dependence_gap"])
                epoch_latent_real_acc_values.append(gap_summary["real_token_accuracy"])
                epoch_latent_shuffle_acc_values.append(gap_summary["shuffled_token_accuracy"])

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
        avg_latent_gap = float(np.mean(epoch_latent_gap_values)) if epoch_latent_gap_values else 0.0
        avg_latent_real_acc = float(np.mean(epoch_latent_real_acc_values)) if epoch_latent_real_acc_values else 0.0
        avg_latent_shuffle_acc = float(np.mean(epoch_latent_shuffle_acc_values)) if epoch_latent_shuffle_acc_values else 0.0

        # 更新 dead-code streak，并根据配置执行 reinit。
        dead_code_streaks = np.where(epoch_code_counts == 0, dead_code_streaks + 1, 0)
        reinit_summary = maybe_reinitialize_dead_codes(
            codebook=codebook,
            optimizer=optimizer,
            latent_reservoir=epoch_latent_reservoir,
            dead_streaks=dead_code_streaks,
            epoch=epoch,
            config=config,
        )
        if reinit_summary["reinitialized"]:
            dead_code_reinit_events.append(reinit_summary)
            logger.warning(
                "epoch %d 触发 dead-code reinit: codes=%s, num_source_vectors=%d",
                epoch,
                reinit_summary["code_indices"],
                reinit_summary["num_source_vectors"],
            )

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
        latent_dependence_gap_history.append(avg_latent_gap)
        latent_real_acc_history.append(avg_latent_real_acc)
        latent_shuffled_acc_history.append(avg_latent_shuffle_acc)

        if epoch == 1 or epoch % 10 == 0 or epoch == config.phase1_epochs:
            logger.info(
                "Epoch %3d/%d — total_loss=%.4f, rec_loss=%.4f, vq_loss=%.4f, token_acc=%.4f, exact_match=%.4f, perplexity=%.4f, used_codes=%d, latent_gap=%.4f",
                epoch,
                config.phase1_epochs,
                avg_loss,
                avg_rec,
                avg_vq,
                token_accuracy,
                exact_match_rate,
                code_usage_summary["codebook_perplexity"],
                code_usage_summary["used_code_count"],
                avg_latent_gap,
            )

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
        "latent_dependence_gap_history": latent_dependence_gap_history,
        "latent_real_acc_history": latent_real_acc_history,
        "latent_shuffled_acc_history": latent_shuffled_acc_history,
        "dead_code_reinit_events": dead_code_reinit_events,
        "warm_start_summary": warm_start_summary,
        "reward_normalization": reward_norm_stats,
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
                "phase1_reward_norm_mode": config.phase1_reward_norm_mode,
                "phase1_reward_norm_eps": config.phase1_reward_norm_eps,
                "phase1_reward_norm_clip": config.phase1_reward_norm_clip,
                "reward_norm_mean": reward_norm_stats["mean"],
                "reward_norm_std": reward_norm_stats["std"],
                "phase1_warm_start_codebook": config.phase1_warm_start_codebook,
                "phase1_dead_code_reinit": config.phase1_dead_code_reinit,
                "phase1_dead_code_patience": config.phase1_dead_code_patience,
                "phase1_dead_code_min_epoch": config.phase1_dead_code_min_epoch,
                "phase1_dead_code_max_codes": config.phase1_dead_code_max_codes,
                "phase1_monitor_latent_dependence": config.phase1_monitor_latent_dependence,
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
    logger.info("Phase I debug branch 训练完成: pair=%s", pair)
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
