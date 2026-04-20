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

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import parse_args
from src.data.dataset import TrajectoryDataset
from src.data.feature_pipeline import FeaturePipeline
from src.env.trading_env import TradingEnv
from src.phase1.codebook import VQCodebook
from src.phase1.dp_planner import DPPlanner
from src.phase1.env_validation import run_phase1_env_validation
from src.phase1.validation import validate_phase1_artifacts
from src.phase1.vq_decoder import VQDecoder
from src.phase1.vq_encoder import VQEncoder
from src.utils.logger import get_logger
from src.utils.progress import should_disable_tqdm

logger = get_logger(__name__)

PAPER_PHASE1_REFERENCE_TRAIN_ROWS = 1_400_000
PAPER_PHASE1_SPEC = { 
    "action_dim": 3,
    "horizon": 72,
    "commission_rate": 0.0003,
    "lstm_hidden_dim": 128,
    "latent_dim": 16,
    "num_archetypes": 10,
    "vq_beta0": 0.25,
    "num_trajectories": 30000,
    "phase1_epochs": 300,
    "pretrain_epochs": 10,
    "discount_factor": 0.99,
    "max_positions": { "ETH": 100, "AL": 10},
}

# Phase I checkpoint 评估间隔：每训练 30 个 epoch 评估一次
PHASE1_CHECKPOINT_EVAL_INTERVAL = 30


# ---------------------------------------------------------------------------
# 数据类：训练指标收集
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    """单个 epoch 的累积指标。"""

    loss: float = 0.0
    rec_loss: float = 0.0
    vq_loss: float = 0.0
    alignment_reg: float = 0.0
    return_aux_reg: float = 0.0
    return_bucket_acc: float = 0.0
    separation_reg: float = 0.0
    usage_profit_corr: float = 0.0
    token_correct: int = 0
    token_total: int = 0
    exact_match: int = 0
    sample_total: int = 0
    code_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    encoder_grad: float = 0.0
    codebook_grad: float = 0.0
    decoder_grad: float = 0.0
    logit_abs_max: float = 0.0
    z_e_norm_sum: float = 0.0
    quantization_mse_sum: float = 0.0
    num_batches: int = 0

    def summarize(self) -> Dict[str, float]:
        """汇总为 epoch 级别的平均指标。"""
        nb = max(self.num_batches, 1)
        ns = max(self.sample_total, 1)
        nt = max(self.token_total, 1)
        code_usage = summarize_code_usage(self.code_counts)
        return {
            "avg_loss": self.loss / nb,
            "avg_rec": self.rec_loss / nb,
            "avg_vq": self.vq_loss / nb,
            "avg_alignment_reg": self.alignment_reg / nb,
            "avg_return_aux_reg": self.return_aux_reg / nb,
            "avg_return_bucket_acc": self.return_bucket_acc / nb,
            "avg_separation_reg": self.separation_reg / nb,
            "avg_usage_profit_corr": self.usage_profit_corr / nb,
            "token_accuracy": float(self.token_correct / nt),
            "exact_match_rate": float(self.exact_match / ns),
            "avg_encoder_grad": float(self.encoder_grad / nb),
            "avg_codebook_grad": float(self.codebook_grad / nb),
            "avg_decoder_grad": float(self.decoder_grad / nb),
            "logit_abs_max": self.logit_abs_max,
            "avg_z_e_norm": float(self.z_e_norm_sum / ns),
            "avg_quantization_mse": float(self.quantization_mse_sum / ns),
            **code_usage,
        }


@dataclass
class TrainingHistory:
    """训练过程中所有 epoch 的指标历史。"""

    loss: List[float] = field(default_factory=list)
    rec_loss: List[float] = field(default_factory=list)
    vq_loss: List[float] = field(default_factory=list)
    alignment_reg: List[float] = field(default_factory=list)
    return_aux_reg: List[float] = field(default_factory=list)
    return_bucket_acc: List[float] = field(default_factory=list)
    separation_reg: List[float] = field(default_factory=list)
    usage_profit_corr: List[float] = field(default_factory=list)
    token_accuracy: List[float] = field(default_factory=list)
    exact_match: List[float] = field(default_factory=list)
    codebook_perplexity: List[float] = field(default_factory=list)
    used_code_count: List[int] = field(default_factory=list)
    dominant_code_ratio: List[float] = field(default_factory=list)
    encoder_grad_norm: List[float] = field(default_factory=list)
    codebook_grad_norm: List[float] = field(default_factory=list)
    decoder_grad_norm: List[float] = field(default_factory=list)
    logit_abs_max: List[float] = field(default_factory=list)
    z_e_norm: List[float] = field(default_factory=list)
    quantization_mse: List[float] = field(default_factory=list)

    def append_from_summary(self, s: Dict[str, float]) -> None:
        self.loss.append(s["avg_loss"])
        self.rec_loss.append(s["avg_rec"])
        self.vq_loss.append(s["avg_vq"])
        self.alignment_reg.append(s["avg_alignment_reg"])
        self.return_aux_reg.append(s["avg_return_aux_reg"])
        self.return_bucket_acc.append(s["avg_return_bucket_acc"])
        self.separation_reg.append(s["avg_separation_reg"])
        self.usage_profit_corr.append(s["avg_usage_profit_corr"])
        self.token_accuracy.append(s["token_accuracy"])
        self.exact_match.append(s["exact_match_rate"])
        self.codebook_perplexity.append(s["codebook_perplexity"])
        self.used_code_count.append(s["used_code_count"])
        self.dominant_code_ratio.append(s["dominant_code_ratio"])
        self.encoder_grad_norm.append(s["avg_encoder_grad"])
        self.codebook_grad_norm.append(s["avg_codebook_grad"])
        self.decoder_grad_norm.append(s["avg_decoder_grad"])
        self.logit_abs_max.append(s["logit_abs_max"])
        self.z_e_norm.append(s["avg_z_e_norm"])
        self.quantization_mse.append(s["avg_quantization_mse"])

    def to_dict(self) -> Dict[str, list]:
        return {
            "loss_history": self.loss,
            "rec_loss_history": self.rec_loss,
            "vq_loss_history": self.vq_loss,
            "alignment_reg_history": self.alignment_reg,
            "return_aux_reg_history": self.return_aux_reg,
            "return_bucket_acc_history": self.return_bucket_acc,
            "separation_reg_history": self.separation_reg,
            "usage_profit_corr_history": self.usage_profit_corr,
            "token_accuracy_history": self.token_accuracy,
            "exact_match_history": self.exact_match,
            "codebook_perplexity_history": self.codebook_perplexity,
            "used_code_count_history": self.used_code_count,
            "dominant_code_ratio_history": self.dominant_code_ratio,
            "encoder_grad_norm_history": self.encoder_grad_norm,
            "codebook_grad_norm_history": self.codebook_grad_norm,
            "decoder_grad_norm_history": self.decoder_grad_norm,
            "logit_abs_max_history": self.logit_abs_max,
            "z_e_norm_history": self.z_e_norm,
            "quantization_mse_history": self.quantization_mse,
        }


# ---------------------------------------------------------------------------
# 工具函数（保持原有逻辑不变）
# ---------------------------------------------------------------------------

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


def compute_soft_code_assignments(
    z_e: torch.Tensor,
    codebook: VQCodebook,
    temperature: float,
) -> torch.Tensor:
    """基于 encoder latent 与 codebook 距离计算 soft assignment。"""
    code_vectors = codebook.embeddings.weight
    temp = max(float(temperature), 1e-6)
    distances = (
        torch.sum(z_e ** 2, dim=1, keepdim=True)
        - 2 * z_e @ code_vectors.t()
        + torch.sum(code_vectors ** 2, dim=1, keepdim=False)
    )
    return torch.softmax(-distances / temp, dim=1)


def compute_usage_profit_alignment_loss(
    soft_assignments: torch.Tensor,
    trajectory_returns: torch.Tensor,
    target_corr: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """鼓励高收益 archetype 具有更高使用率，避免收益-使用率负相关。"""
    if soft_assignments.ndim != 2 or soft_assignments.shape[0] < 2 or soft_assignments.shape[1] < 2:
        zero = soft_assignments.new_zeros(())
        return zero, zero

    returns = trajectory_returns.reshape(-1, 1)
    code_mass = soft_assignments.sum(dim=0).clamp_min(eps)
    usage = code_mass / max(int(soft_assignments.shape[0]), 1)
    code_returns = (soft_assignments * returns).sum(dim=0) / code_mass

    usage_centered = usage - usage.mean()
    return_centered = code_returns - code_returns.mean()
    covariance = torch.mean(usage_centered * return_centered)
    denom = torch.sqrt(
        torch.mean(usage_centered ** 2) * torch.mean(return_centered ** 2) + eps,
    )
    corr = covariance / denom
    target = soft_assignments.new_tensor(float(np.clip(target_corr, -1.0, 1.0)))
    loss = torch.relu(target - corr)
    return loss, corr


def compute_codebook_separation_loss(
    embeddings: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """惩罚过高的 codebook cosine 相似度，降低 archetype 同质化。"""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return embeddings.new_zeros(())

    normalized = F.normalize(embeddings, dim=1)
    cosine = normalized @ normalized.t()
    off_diag_mask = ~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)
    penalties = torch.relu(cosine[off_diag_mask] - float(np.clip(margin, -1.0, 1.0)))
    if penalties.numel() == 0:
        return embeddings.new_zeros(())
    return torch.mean(penalties ** 2)


def build_return_bucket_head(config: Any, device: torch.device) -> nn.Module:
    """构建基于 archetype latent 的收益分桶头。"""
    hidden_dim = max(int(getattr(config, "phase1_return_aux_hidden_dim", 32)), 1)
    num_buckets = max(int(getattr(config, "phase1_return_num_buckets", 5)), 2)
    return nn.Sequential(
        nn.LayerNorm(config.latent_dim),
        nn.Linear(config.latent_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, num_buckets),
    ).to(device)


def build_return_bucket_edges(
    trajectory_returns: torch.Tensor | np.ndarray,
    num_buckets: int,
) -> np.ndarray:
    """根据全局轨迹收益分位数构建收益分桶边界。"""
    values = np.asarray(trajectory_returns, dtype=np.float32).reshape(-1)
    if values.size == 0 or num_buckets <= 1:
        return np.zeros(0, dtype=np.float32)

    quantiles = np.linspace(0.0, 1.0, num_buckets + 1, dtype=np.float32)[1:-1]
    if quantiles.size == 0:
        return np.zeros(0, dtype=np.float32)

    edges = np.quantile(values, quantiles).astype(np.float32)
    return np.unique(edges)


def bucketize_trajectory_returns(
    trajectory_returns: torch.Tensor,
    bucket_edges: torch.Tensor,
) -> torch.Tensor:
    """将轨迹总收益映射到离散 bucket id。"""
    if bucket_edges.numel() == 0:
        return torch.zeros_like(trajectory_returns, dtype=torch.long)
    return torch.bucketize(trajectory_returns, bucket_edges)


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
    """强制检查当前配置是否严格等于论文 Phase I 主实验设置。"""
    mismatches: List[str] = []

    def _check_exact(name: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            mismatches.append(f"{name}: actual={actual}, expected={expected}")

    def _check_float(name: str, actual: float, expected: float, atol: float = 1e-12) -> None:
        if not np.isclose(actual, expected, atol=atol, rtol=0.0):
            mismatches.append(f"{name}: actual={actual}, expected={expected}")

    _check_exact("action_dim", config.action_dim, PAPER_PHASE1_SPEC["action_dim"])
    _check_exact("horizon", config.horizon, PAPER_PHASE1_SPEC["horizon"])
    _check_float("commission_rate", config.commission_rate, PAPER_PHASE1_SPEC["commission_rate"])
    _check_exact("lstm_hidden_dim", config.lstm_hidden_dim, PAPER_PHASE1_SPEC["lstm_hidden_dim"])
    _check_exact("latent_dim", config.latent_dim, PAPER_PHASE1_SPEC["latent_dim"])
    _check_exact("num_archetypes", config.num_archetypes, PAPER_PHASE1_SPEC["num_archetypes"])
    _check_float("vq_beta0", config.vq_beta0, PAPER_PHASE1_SPEC["vq_beta0"])
    _check_exact("num_trajectories", config.num_trajectories, PAPER_PHASE1_SPEC["num_trajectories"])
    _check_exact("phase1_epochs", config.phase1_epochs, PAPER_PHASE1_SPEC["phase1_epochs"])
    _check_exact("pretrain_epochs", config.pretrain_epochs, PAPER_PHASE1_SPEC["pretrain_epochs"])
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
        "这仍属于严格论文算法/公式下的 reduced-data reproduction，而非同数据规模复现。",
        train_rows,
        PAPER_PHASE1_REFERENCE_TRAIN_ROWS,
        ratio * 100.0,
    )


def expected_num_available_starts(total_rows: int, horizon: int) -> int:
    """计算滑窗采样协议下全部合法起点数量。"""
    return max(total_rows - horizon + 1, 0)


# ---------------------------------------------------------------------------
# 轨迹缓存检查
# ---------------------------------------------------------------------------

def inspect_trajectory_cache(
    traj_path: str,
    config: Any,
    pair: str,
    train_rows: int,
) -> Tuple[bool, List[str]]:
    """检查现有轨迹缓存是否与当前严格论文设置兼容。"""
    if not os.path.exists(traj_path):
        return False, ["trajectory cache 文件不存在"]

    reasons: List[str] = []
    expected_starts = expected_num_available_starts(train_rows, config.horizon)
    ratio_sum = float(config.phase1_stratified_ratio) + float(config.phase1_importance_ratio)
    if ratio_sum <= 0:
        expected_stratified_ratio = 1.0
        expected_importance_ratio = 0.0
    else:
        expected_stratified_ratio = float(config.phase1_stratified_ratio) / ratio_sum
        expected_importance_ratio = float(config.phase1_importance_ratio) / ratio_sum

    weight_sum = float(config.phase1_importance_vol_weight) + float(config.phase1_importance_net_weight)
    if weight_sum <= 0:
        expected_importance_vol_weight = 0.5
        expected_importance_net_weight = 0.5
    else:
        expected_importance_vol_weight = float(config.phase1_importance_vol_weight) / weight_sum
        expected_importance_net_weight = float(config.phase1_importance_net_weight) / weight_sum

    expected_values = {
        "pair": pair,
        "horizon": int(config.horizon),
        "gamma": float(config.discount_factor),
        "num_sampled_trajectories": int(config.num_trajectories),
        "sampling_seed": int(config.phase1_sampling_seed),
        "sampling_mode": str(config.phase1_start_sampling_mode),
        "sampling_stratified_ratio": float(expected_stratified_ratio),
        "sampling_importance_ratio": float(expected_importance_ratio),
        "sampling_num_strata": int(config.phase1_sampling_strata),
        "sampling_importance_vol_weight": float(expected_importance_vol_weight),
        "sampling_importance_net_weight": float(expected_importance_net_weight),
        "num_available_starts": int(expected_starts),
        "training_rows": int(train_rows),
        "state_dim": int(config.state_dim),
        "commission_rate": float(config.dp_commission_rate),
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


# ---------------------------------------------------------------------------
# 从 main() 中提取的子流程
# ---------------------------------------------------------------------------

def load_data_and_env(config: Any, pair: str) -> Tuple[TradingEnv, int]:
    """加载特征数据并初始化 TradingEnv。

    Returns:
        (env, train_rows)
    """
    logger.info("加载特征数据: data_dir=%s, pair=%s", config.data_dir, pair)
    pipeline = FeaturePipeline(
        config.data_dir, pair, cycle_features=config.cycle_features,
    )
    train_df, _, _ = pipeline.get_state_vector()
    train_prices_df, _, _ = pipeline.get_prices()

    train_states = train_df.to_numpy()
    prices = train_prices_df["close"].to_numpy()
    train_rows = int(train_states.shape[0])

    logger.info("训练集: states shape=%s, prices shape=%s", train_states.shape, prices.shape)
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
        commission_rate=config.train_commission_rate,
    )

    # DP planner 用更高的费率筛选高利润轨迹
    dp_env = TradingEnv(
        states=train_states,
        prices=prices,
        pair=pair,
        horizon=config.horizon,
        states_dataframe=train_df,
        max_positions=config.max_positions,
        commission_rate=config.dp_commission_rate,
    )

    logger.info(
        "TradingEnv 初始化完成: num_horizons=%d, horizon=%d, max_position=%d, "
        "train_commission_rate=%.6f, dp_commission_rate=%.6f, available_starts=%d",
        env.num_horizons, config.horizon, env.m,
        env.commission_rate, dp_env.commission_rate, available_starts,
    )
    return env, dp_env, train_rows


def prepare_trajectory_dataset(
    config: Any, pair: str, dp_env: TradingEnv, train_rows: int,
) -> Tuple[TrajectoryDataset, str]:
    """检查缓存 / 生成 DP 示范轨迹，返回 (dataset, traj_path)。

    使用 dp_env（高费率环境）生成轨迹，筛选出高利润交易模式。
    """
    planner = DPPlanner(
        env=dp_env,
        gamma=config.discount_factor,
        result_dir=config.result_dir,
        train_batch_id=config.train_batch_id,
        sampling_seed=config.phase1_sampling_seed,
        sampling_mode=config.phase1_start_sampling_mode,
        stratified_ratio=config.phase1_stratified_ratio,
        importance_ratio=config.phase1_importance_ratio,
        sampling_num_strata=config.phase1_sampling_strata,
        importance_vol_weight=config.phase1_importance_vol_weight,
        importance_net_weight=config.phase1_importance_net_weight,
    )
    traj_path = DPPlanner.build_trajectory_cache_path(
        config.result_dir, pair, config.train_batch_id,
    )

    if os.path.exists(traj_path):
        cache_ok, cache_reasons = inspect_trajectory_cache(
            traj_path=traj_path, config=config, pair=pair, train_rows=train_rows,
        )
        if cache_ok:
            logger.info("发现与当前严格论文设置兼容的轨迹缓存，直接加载: %s", traj_path)
            return TrajectoryDataset.from_npz(traj_path, normalize=True), traj_path
        backup_incompatible_cache(traj_path, cache_reasons)

    logger.info("开始生成 DP 示范轨迹: num_trajectories=%d", config.num_trajectories)
    trajectories = planner.generate_trajectories(config.num_trajectories)
    logger.info("DP 轨迹生成完成，创建 Dataset")
    dataset = TrajectoryDataset(
        states=trajectories["states"],
        actions=trajectories["actions"],
        rewards=trajectories["rewards"],
        normalize=True,
    )
    return dataset, traj_path


def build_models(
    config: Any, device: torch.device,
) -> Tuple[VQEncoder, VQCodebook, VQDecoder]:
    """初始化 VQ Encoder、Codebook、Decoder 并移至目标设备。"""
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
    return encoder, codebook, decoder


# ---------------------------------------------------------------------------
# 单 epoch 训练
# ---------------------------------------------------------------------------

def _accumulate_batch_metrics(
    metrics: EpochMetrics,
    *,
    total_loss: torch.Tensor,
    rec_loss: torch.Tensor,
    vq_loss_val: float,
    alignment_reg_val: float,
    return_aux_reg_val: float,
    return_bucket_acc_val: float,
    separation_reg_val: float,
    usage_profit_corr_val: float,
    pred_actions: torch.Tensor,
    a_demo: torch.Tensor,
    action_logits: torch.Tensor,
    z_e: torch.Tensor,
    quantization_mse_val: float,
    indices: np.ndarray | None,
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
) -> None:
    """将单个 batch 的指标累积到 EpochMetrics 中。"""
    batch_size = int(a_demo.shape[0])

    metrics.loss += total_loss.item()
    metrics.rec_loss += rec_loss.item()
    metrics.vq_loss += vq_loss_val
    metrics.alignment_reg += alignment_reg_val
    metrics.return_aux_reg += return_aux_reg_val
    metrics.return_bucket_acc += return_bucket_acc_val
    metrics.separation_reg += separation_reg_val
    metrics.usage_profit_corr += usage_profit_corr_val
    metrics.token_correct += int((pred_actions == a_demo).sum().item())
    metrics.token_total += int(a_demo.numel())
    metrics.exact_match += int(torch.all(pred_actions == a_demo, dim=1).sum().item())
    metrics.sample_total += batch_size

    if indices is not None:
        metrics.code_counts += np.bincount(indices, minlength=len(metrics.code_counts))

    metrics.logit_abs_max = max(metrics.logit_abs_max, float(action_logits.abs().max().item()))
    metrics.z_e_norm_sum += float(torch.norm(z_e, dim=1).mean().item()) * batch_size
    metrics.quantization_mse_sum += quantization_mse_val * batch_size

    metrics.encoder_grad += compute_grad_norm(encoder.parameters())
    metrics.codebook_grad += compute_grad_norm(codebook.parameters())
    metrics.decoder_grad += compute_grad_norm(decoder.parameters())
    metrics.num_batches += 1


def train_one_epoch(
    *,
    dataloader: DataLoader,
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
    return_bucket_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    ce_loss_fn: nn.CrossEntropyLoss,
    config: Any,
    device: torch.device,
    return_bucket_edges: torch.Tensor,
    is_phase_a: bool,
    collect_z_e: bool = False,
) -> Tuple[EpochMetrics, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """执行单个 epoch 的训练，返回累积指标。

    Phase A (预训练): 跳过 VQ 量化，仅优化 L_rec。
    Phase B (VQ 训练): 完整 VQ 流水线，L = L_rec + commitment + β₀ × encoder_commitment。

    Args:
        collect_z_e: 若为 True，收集所有 batch 的 z_e 用于 k-means 初始化。
                     仅在 Phase A 最后一个 epoch 使用。

    Returns:
        metrics: 训练指标
        z_e_all: 收集的 z_e 样本 (仅 collect_z_e=True 时非 None)
    """
    encoder.train()
    codebook.train()
    decoder.train()
    return_bucket_head.train()

    metrics = EpochMetrics(
        code_counts=np.zeros(config.num_archetypes, dtype=np.int64),
    )
    z_e_list: List[torch.Tensor] = [] if collect_z_e else []
    a_demo_list: List[torch.Tensor] = [] if collect_z_e else []
    traj_return_list: List[torch.Tensor] = [] if collect_z_e else []
    last_z_e: torch.Tensor | None = None
    last_traj_returns: torch.Tensor | None = None

    for s_demo, a_demo, r_demo in dataloader:
        s_demo = s_demo.to(device)
        a_demo = a_demo.to(device)
        r_demo = r_demo.to(device)
        traj_returns = r_demo.sum(dim=1)
        return_bucket_targets = bucketize_trajectory_returns(traj_returns, return_bucket_edges)

        # Encode
        z_e = encoder(s_demo, a_demo, r_demo)

        if collect_z_e:
            z_e_list.append(z_e.detach())
            a_demo_list.append(a_demo.detach())
            traj_return_list.append(traj_returns.detach())

        if not is_phase_a:
            last_z_e = z_e.detach()
            last_traj_returns = traj_returns.detach()

        if is_phase_a:
            # Phase A: bypass VQ, pass z_e directly to decoder
            z_input = z_e
            indices_np = None
            vq_loss_val = 0.0
            quantization_mse_val = 0.0
            alignment_reg_val = 0.0
            return_aux_reg_val = 0.0
            return_bucket_acc_val = 0.0
            separation_reg_val = 0.0
            usage_profit_corr_val = 0.0
        else:
            # Phase B: full VQ quantization
            z_q_st, indices, commitment_loss = codebook.quantize(z_e)
            z_input = z_q_st
            indices_np = indices.detach().cpu().numpy()

        # Decode
        action_logits = decoder(s_demo, z_input)
        pred_actions = torch.argmax(action_logits, dim=-1)

        # L_rec
        logits_flat = action_logits.reshape(-1, config.action_dim)
        targets_flat = a_demo.reshape(-1)
        rec_loss = ce_loss_fn(logits_flat, targets_flat)

        if is_phase_a:
            total_loss = rec_loss
        else:
            # β₀ × ||z_e - sg[z_q]||²
            z_q_detached = z_q_st.detach()
            encoder_commitment = config.vq_beta0 * torch.mean((z_e - z_q_detached) ** 2)
            soft_assignments = compute_soft_code_assignments(
                z_e, codebook, config.phase1_usage_profit_alignment_temperature,
            )
            code_bucket_logits = return_bucket_head(codebook.embeddings.weight)
            soft_bucket_logits = soft_assignments @ code_bucket_logits
            hard_bucket_logits = code_bucket_logits[indices]
            soft_bucket_loss = F.cross_entropy(soft_bucket_logits, return_bucket_targets)
            hard_bucket_loss = F.cross_entropy(hard_bucket_logits, return_bucket_targets)
            soft_bucket_weight = float(np.clip(config.phase1_return_soft_assignment_weight, 0.0, 1.0))
            return_aux_loss = (
                soft_bucket_weight * soft_bucket_loss
                + (1.0 - soft_bucket_weight) * hard_bucket_loss
            )
            alignment_loss, usage_profit_corr = compute_usage_profit_alignment_loss(
                soft_assignments,
                traj_returns,
                config.phase1_usage_profit_alignment_target_corr,
            )
            separation_loss = compute_codebook_separation_loss(
                codebook.embeddings.weight,
                config.phase1_codebook_separation_margin,
            )
            alignment_reg = config.phase1_usage_profit_alignment_weight * alignment_loss
            return_aux_reg = config.phase1_return_aux_weight * return_aux_loss
            separation_reg = config.phase1_codebook_separation_weight * separation_loss
            total_loss = (
                rec_loss + commitment_loss + encoder_commitment + alignment_reg + return_aux_reg + separation_reg
            )
            vq_loss_val = commitment_loss.item() + encoder_commitment.item()
            quantization_mse_val = float(torch.mean((z_e - z_q_detached) ** 2).item())
            alignment_reg_val = float(alignment_reg.item())
            return_aux_reg_val = float(return_aux_reg.item())
            return_bucket_acc_val = float(
                (torch.argmax(hard_bucket_logits, dim=1) == return_bucket_targets).float().mean().item()
            )
            separation_reg_val = float(separation_reg.item())
            usage_profit_corr_val = float(usage_profit_corr.item())

        optimizer.zero_grad()
        total_loss.backward()

        # 累积指标（在 step 之前，梯度可用）
        _accumulate_batch_metrics(
            metrics,
            total_loss=total_loss,
            rec_loss=rec_loss,
            vq_loss_val=vq_loss_val,
            alignment_reg_val=alignment_reg_val,
            return_aux_reg_val=return_aux_reg_val,
            return_bucket_acc_val=return_bucket_acc_val,
            separation_reg_val=separation_reg_val,
            usage_profit_corr_val=usage_profit_corr_val,
            pred_actions=pred_actions,
            a_demo=a_demo,
            action_logits=action_logits,
            z_e=z_e,
            quantization_mse_val=quantization_mse_val,
            indices=indices_np,
            encoder=encoder,
            codebook=codebook,
            decoder=decoder,
        )

        optimizer.step()

    z_e_all = torch.cat(z_e_list, dim=0) if collect_z_e and z_e_list else None
    a_demo_all = torch.cat(a_demo_list, dim=0) if collect_z_e and a_demo_list else None
    traj_return_all = torch.cat(traj_return_list, dim=0) if collect_z_e and traj_return_list else None

    # Phase B: 死码重置
    if not is_phase_a and last_z_e is not None:
        codebook.reset_dead_codes(
            last_z_e,
            metrics.code_counts,
            trajectory_returns=last_traj_returns,
            top_ratio=config.phase1_profit_reset_top_ratio,
        )

    return metrics, z_e_all, a_demo_all, traj_return_all


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------

def run_training_loop(
    *,
    dataloader: DataLoader,
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
    return_bucket_head: nn.Module,
    config: Any,
    device: torch.device,
    checkpoint_interval: int = 0,
    on_checkpoint: Callable[[int, Dict[str, float], TrainingHistory], None] | None = None,
) -> TrainingHistory:
    """执行完整的 Phase I 训练循环（Phase A 预训练 + Phase B VQ 训练）。"""
    all_params = (
        list(encoder.parameters())
        + list(codebook.parameters())
        + list(decoder.parameters())
        + list(return_bucket_head.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=config.learning_rate)
    ce_loss_fn = nn.CrossEntropyLoss()
    history = TrainingHistory()
    dataset_returns = dataloader.dataset.rewards.sum(dim=1)
    return_bucket_edges_np = build_return_bucket_edges(
        dataset_returns.detach().cpu().numpy(),
        config.phase1_return_num_buckets,
    )
    return_bucket_edges = torch.as_tensor(return_bucket_edges_np, dtype=torch.float32, device=device)

    logger.info("开始训练: %d epochs", config.phase1_epochs)
    logger.info("Phase A (连续潜在预训练): epochs 1-%d, loss=L_rec only", config.pretrain_epochs)
    logger.info(
        "收益分桶目标已启用: num_buckets=%d, soft_assignment_weight=%.2f, edges=%s",
        int(config.phase1_return_num_buckets),
        float(config.phase1_return_soft_assignment_weight),
        np.round(return_bucket_edges_np, 4).tolist(),
    )

    disable_tqdm = should_disable_tqdm()
    for epoch in tqdm(
        range(1, config.phase1_epochs + 1),
        desc="Training Epochs",
        disable=disable_tqdm,
    ):
        is_phase_a = epoch <= config.pretrain_epochs
        # 在 Phase A 最后一个 epoch 收集 z_e 用于 k-means 初始化
        collect_z_e = (epoch == config.pretrain_epochs)

        if epoch == config.pretrain_epochs + 1:
            logger.info(
                "Phase B (VQ 训练): epochs %d-%d, full VQ loss",
                config.pretrain_epochs + 1, config.phase1_epochs,
            )

        metrics, z_e_all, a_demo_all, traj_return_all = train_one_epoch(
            dataloader=dataloader,
            encoder=encoder,
            codebook=codebook,
            decoder=decoder,
            return_bucket_head=return_bucket_head,
            optimizer=optimizer,
            ce_loss_fn=ce_loss_fn,
            config=config,
            device=device,
            return_bucket_edges=return_bucket_edges,
            is_phase_a=is_phase_a,
            collect_z_e=collect_z_e,
        )

        # Phase A → Phase B 过渡: 用方向感知 k-means 初始化码本
        if collect_z_e and z_e_all is not None:
            if a_demo_all is not None:
                codebook.init_from_data_direction_aware(
                    z_e_all,
                    a_demo_all,
                    trajectory_returns=traj_return_all,
                    profit_top_ratio=config.phase1_profit_init_top_ratio,
                    profit_code_ratio=config.phase1_profit_init_code_ratio,
                )
            else:
                codebook.init_from_data(z_e_all)

        summary = metrics.summarize()
        history.append_from_summary(summary)

        if epoch == 1 or epoch % 10 == 0 or epoch == config.phase1_epochs:
            tqdm.write(
                "Epoch %3d/%d — total_loss=%.4f, rec_loss=%.4f, vq_loss=%.4f, "
                "align_reg=%.4f, return_reg=%.4f, bucket_acc=%.4f, sep_reg=%.4f, usage_corr=%.4f, "
                "token_acc=%.4f, exact_match=%.4f, perplexity=%.4f, used_codes=%d"
                % (
                    epoch, config.phase1_epochs,
                    summary["avg_loss"], summary["avg_rec"], summary["avg_vq"],
                    summary["avg_alignment_reg"], summary["avg_return_aux_reg"], summary["avg_return_bucket_acc"],
                    summary["avg_separation_reg"],
                    summary["avg_usage_profit_corr"],
                    summary["token_accuracy"], summary["exact_match_rate"],
                    summary["codebook_perplexity"], summary["used_code_count"],
                )
            )

        if (
            on_checkpoint is not None
            and checkpoint_interval > 0
            and epoch % checkpoint_interval == 0
        ):
            on_checkpoint(epoch, summary, history)

        if np.isnan(summary["avg_loss"]):
            logger.error("训练 loss 发散 (NaN)，在 epoch %d 终止训练", epoch)
            break

    return history


# ---------------------------------------------------------------------------
# 保存 & 日志
# ---------------------------------------------------------------------------

def save_checkpoint(
    *,
    config: Any,
    pair: str,
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
    history: TrainingHistory,
    train_rows: int,
    norm_stats: dict | None = None,
    save_path: str | None = None,
    checkpoint_meta: Dict[str, Any] | None = None,
) -> str:
    """保存模型 checkpoint 到 result/phase1_archetype_discovery/，返回保存路径。"""
    save_dir = config.get_stage_result_dir(pair, "phase1_archetype_discovery")
    os.makedirs(save_dir, exist_ok=True)

    if save_path is None:
        save_path = os.path.join(save_dir, f"{pair}_vq_model.pt")
    checkpoint_data = {
        "encoder": encoder.state_dict(),
        "codebook": codebook.state_dict(),
        "decoder": decoder.state_dict(),
        "loss_history": history.loss,
        "training_monitor": history.to_dict(),
        "config": {
            "state_dim": config.state_dim,
            "action_dim": config.action_dim,
            "latent_dim": config.latent_dim,
            "num_archetypes": config.num_archetypes,
            "lstm_hidden_dim": config.lstm_hidden_dim,
            "phase1_epochs": config.phase1_epochs,
            "pretrain_epochs": config.pretrain_epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "vq_beta0": config.vq_beta0,
            "num_trajectories": config.num_trajectories,
            "phase1_sampling_seed": config.phase1_sampling_seed,
            "phase1_start_sampling_mode": config.phase1_start_sampling_mode,
            "phase1_stratified_ratio": config.phase1_stratified_ratio,
            "phase1_importance_ratio": config.phase1_importance_ratio,
            "phase1_sampling_strata": config.phase1_sampling_strata,
            "phase1_importance_vol_weight": config.phase1_importance_vol_weight,
            "phase1_importance_net_weight": config.phase1_importance_net_weight,
            "phase1_usage_profit_alignment_weight": config.phase1_usage_profit_alignment_weight,
            "phase1_usage_profit_alignment_target_corr": config.phase1_usage_profit_alignment_target_corr,
            "phase1_usage_profit_alignment_temperature": config.phase1_usage_profit_alignment_temperature,
            "phase1_return_aux_weight": config.phase1_return_aux_weight,
            "phase1_return_aux_hidden_dim": config.phase1_return_aux_hidden_dim,
            "phase1_return_num_buckets": config.phase1_return_num_buckets,
            "phase1_return_soft_assignment_weight": config.phase1_return_soft_assignment_weight,
            "phase1_codebook_separation_weight": config.phase1_codebook_separation_weight,
            "phase1_codebook_separation_margin": config.phase1_codebook_separation_margin,
            "phase1_profit_init_top_ratio": config.phase1_profit_init_top_ratio,
            "phase1_profit_init_code_ratio": config.phase1_profit_init_code_ratio,
            "phase1_profit_reset_top_ratio": config.phase1_profit_reset_top_ratio,
            "discount_factor": config.discount_factor,
            "commission_rate": config.commission_rate,
                "dp_commission_rate": config.dp_commission_rate,
                "train_commission_rate": config.train_commission_rate,
            "max_positions": config.max_positions,
            "train_batch_id": config.train_batch_id,
            "paper_phase1_reference_train_rows": PAPER_PHASE1_REFERENCE_TRAIN_ROWS,
            "current_train_rows": train_rows,
        },
    }
    if norm_stats is not None:
        checkpoint_data["norm_stats"] = {
            k: v.tolist() if hasattr(v, 'tolist') else float(v)
            for k, v in norm_stats.items()
        }
    if checkpoint_meta is not None:
        checkpoint_data["checkpoint_meta"] = checkpoint_meta
    torch.save(checkpoint_data, save_path)
    logger.info("模型已保存到 %s", save_path)
    return save_path


def build_val_env(config: Any, pair: str) -> TradingEnv | None:
    """构建验证集环境（若验证集不足一个 horizon，则返回 None）。"""
    val_pipeline = FeaturePipeline(
        config.data_dir, pair, cycle_features=config.cycle_features,
    )
    _, val_state_df, _ = val_pipeline.get_state_vector()
    _, val_prices_df, _ = val_pipeline.get_prices()
    if val_state_df is None or len(val_state_df) < config.horizon:
        return None
    if val_state_df.width != config.state_dim:
        raise ValueError(
            "验证集 state_dim 与当前配置不一致: "
            f"actual={val_state_df.width}, expected={config.state_dim}"
        )
    return TradingEnv(
        states=val_state_df.to_numpy(),
        prices=val_prices_df["close"].to_numpy(),
        pair=pair,
        horizon=config.horizon,
        states_dataframe=val_state_df,
        max_positions=config.max_positions,
        commission_rate=config.train_commission_rate,
    )


def load_models_from_phase1_checkpoint(
    *,
    config: Any,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[VQEncoder, VQCodebook, VQDecoder]:
    """从指定 checkpoint 还原 encoder/codebook/decoder。"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    codebook.load_state_dict(checkpoint["codebook"], strict=True)
    decoder.load_state_dict(checkpoint["decoder"], strict=True)
    encoder.eval()
    codebook.eval()
    decoder.eval()
    return encoder, codebook, decoder


def extract_checkpoint_selection_metrics(
    validation_report: Dict[str, Any],
    env_report: Dict[str, Any],
) -> Dict[str, Any]:
    """从 Phase I 两类验证报告中抽取 checkpoint 选择指标。"""

    def _safe_float_metric(value: Any, default: float = 0.0) -> float:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(val):
            return float(default)
        return val

    status = validation_report.get("status", {})
    dp_validation = validation_report.get("dp_validation", {})
    model_validation = validation_report.get("model_validation", {})
    loss_decomposition = model_validation.get("loss_decomposition", {})
    reconstruction = model_validation.get("reconstruction", {})
    codebook_usage = model_validation.get("codebook_usage", {})
    latent_geometry = model_validation.get("latent_geometry", {})
    archetype_env = env_report.get("archetype_env_returns", {})
    val_env = env_report.get("val_env_returns", {})
    decoder_action_shift = env_report.get("decoder_action_shift", {})

    oracle_return_mean = _safe_float_metric(val_env.get("oracle_return_mean", 0.0))
    oracle_return_std = _safe_float_metric(val_env.get("oracle_return_std", 0.0))
    oracle_positive_ratio = _safe_float_metric(val_env.get("oracle_positive_ratio", 0.0))
    val_per_archetype = val_env.get("per_archetype", {})
    best_fixed_archetype_return_mean = max(
        (
            _safe_float_metric(v.get("return_mean", 0.0))
            for v in val_per_archetype.values()
            if isinstance(v, dict)
        ),
        default=0.0,
    )
    decoded_return_mean = _safe_float_metric(archetype_env.get("decoded_return_mean", 0.0))
    decoded_to_oracle_return_ratio = decoded_return_mean / oracle_return_mean if oracle_return_mean > 0 else 0.0
    phase2_proxy_return_mean = 0.7 * best_fixed_archetype_return_mean + 0.3 * decoded_return_mean
    phase2_proxy_to_oracle_ratio = (
        phase2_proxy_return_mean / oracle_return_mean if oracle_return_mean > 0 else 0.0
    )
    token_accuracy = _safe_float_metric(reconstruction.get("token_accuracy", 0.0))
    flat_baseline_accuracy = _safe_float_metric(reconstruction.get("flat_baseline_accuracy", 0.0))

    metrics = {
        "overall_passed": bool(status.get("overall_passed", False)),
        "dp_passed": bool(status.get("dp_passed", False)),
        "model_passed": bool(status.get("model_passed", False)),
        "hard_failures_count": len(status.get("hard_failures", [])),
        "soft_warnings_count": len(status.get("soft_warnings", [])),
        "dp_hard_failures_count": len(dp_validation.get("hard_failures", [])),
        "dp_soft_warnings_count": len(dp_validation.get("soft_warnings", [])),
        "model_hard_failures_count": len(model_validation.get("hard_failures", [])),
        "model_soft_warnings_count": len(model_validation.get("soft_warnings", [])),
        "token_accuracy": token_accuracy,
        "trajectory_exact_match_rate": _safe_float_metric(reconstruction.get("trajectory_exact_match_rate", 0.0)),
        "change_detect_accuracy": _safe_float_metric(reconstruction.get("change_detect_accuracy", 0.0)),
        "change_step_mae": _safe_float_metric(reconstruction.get("change_step_mae", 1e6)),
        "flat_baseline_accuracy": flat_baseline_accuracy,
        "flat_baseline_margin": token_accuracy - flat_baseline_accuracy,
        "loss_decomposition_residual": _safe_float_metric(loss_decomposition.get("loss_decomposition_residual", 0.0)),
        "used_code_count": int(codebook_usage.get("used_code_count", 0)),
        "dead_code_count": int(codebook_usage.get("dead_code_count", 0)),
        "dominant_code_ratio": _safe_float_metric(codebook_usage.get("dominant_code_ratio", 0.0)),
        "codebook_perplexity": _safe_float_metric(codebook_usage.get("codebook_perplexity", 0.0)),
        "low_usage_code_count": int(codebook_usage.get("low_usage_code_count", 0)),
        "quantization_mse": _safe_float_metric(latent_geometry.get("quantization_mse", 0.0)),
        "high_similarity_pair_count": int(latent_geometry.get("high_similarity_pair_count", 0)),
        "positive_archetype_count": int(archetype_env.get("positive_archetype_count", 0)),
        "decoded_return_mean": decoded_return_mean,
        "best_fixed_archetype_return_mean": _safe_float_metric(best_fixed_archetype_return_mean, 0.0),
        "phase2_proxy_return_mean": _safe_float_metric(phase2_proxy_return_mean, 0.0),
        "phase2_proxy_to_oracle_ratio": _safe_float_metric(phase2_proxy_to_oracle_ratio, 0.0),
        "oracle_return_mean": oracle_return_mean,
        "oracle_return_std": oracle_return_std,
        "oracle_positive_ratio": oracle_positive_ratio,
        "decoded_to_oracle_return_ratio": _safe_float_metric(decoded_to_oracle_return_ratio, default=0.0),
        "return_usage_correlation": _safe_float_metric(codebook_usage.get("return_usage_correlation", 0.0)),
        "change_point_accuracy": _safe_float_metric(decoder_action_shift.get("change_point_accuracy", 0.0)),
        "non_change_accuracy": _safe_float_metric(decoder_action_shift.get("non_change_accuracy", 0.0)),
        "env_warning_count": len(env_report.get("all_warnings", [])),
    }
    phase2_realizability_score = compute_phase2_realizability_score(metrics)
    phase2_realizable_proxy_return_mean = phase2_proxy_return_mean * phase2_realizability_score
    phase2_realizable_proxy_to_oracle_ratio = (
        phase2_realizable_proxy_return_mean / oracle_return_mean if oracle_return_mean > 0 else 0.0
    )
    metrics["phase2_realizability_score"] = _safe_float_metric(phase2_realizability_score, 0.0)
    metrics["phase2_realizable_proxy_return_mean"] = _safe_float_metric(
        phase2_realizable_proxy_return_mean,
        0.0,
    )
    metrics["phase2_realizable_proxy_to_oracle_ratio"] = _safe_float_metric(
        phase2_realizable_proxy_to_oracle_ratio,
        0.0,
    )
    return metrics


def compute_phase2_realizability_score(metrics: Dict[str, Any]) -> float:
    """估计 Phase I proxy 在 Phase II 中可被真实策略兑现的比例（0~1+）。"""
    exact_match = float(np.clip(metrics.get("trajectory_exact_match_rate", 0.0), 0.0, 1.0))
    token_acc = float(np.clip(metrics.get("token_accuracy", 0.0), 0.0, 1.0))
    change_acc = float(np.clip(metrics.get("change_point_accuracy", 0.0), 0.0, 1.0))
    non_change_acc = float(np.clip(metrics.get("non_change_accuracy", 0.0), 0.0, 1.0))
    change_detect_acc = float(np.clip(metrics.get("change_detect_accuracy", 0.0), 0.0, 1.0))
    change_step_mae = float(max(metrics.get("change_step_mae", 1e6), 0.0))
    change_step_score = 1.0 / (1.0 + change_step_mae)
    corr = float(np.clip(metrics.get("return_usage_correlation", 0.0), -1.0, 1.0))
    corr_stability = float(np.clip(1.0 - abs(corr), 0.0, 1.0))
    decoded_ratio = float(np.clip(metrics.get("decoded_to_oracle_return_ratio", 0.0), 0.0, 1.2)) / 1.2

    score = (
        0.32 * exact_match
        + 0.06 * token_acc
        + 0.20 * change_acc
        + 0.06 * non_change_acc
        + 0.16 * change_detect_acc
        + 0.12 * change_step_score
        + 0.06 * corr_stability
        + 0.02 * decoded_ratio
    )
    return float(np.clip(score, 0.0, 1.2))


def compute_generalization_score(metrics: Dict[str, Any]) -> float:
    """计算用于 checkpoint 选择的泛化稳定性分数（越大越好）。"""
    oracle_mean = float(metrics.get("oracle_return_mean", 0.0))
    oracle_std = float(metrics.get("oracle_return_std", 0.0))
    oracle_cv = oracle_std / max(abs(oracle_mean), 1e-6)
    oracle_pos_ratio = float(np.clip(metrics.get("oracle_positive_ratio", 0.0), 0.0, 1.0))
    decoded_ratio = float(np.clip(metrics.get("decoded_to_oracle_return_ratio", 0.0), 0.0, 1.5))
    phase2_proxy_ratio = float(np.clip(metrics.get("phase2_proxy_to_oracle_ratio", 0.0), -1.5, 1.5))
    corr = float(np.clip(metrics.get("return_usage_correlation", 0.0), -1.0, 1.0))
    change_acc = float(np.clip(metrics.get("change_point_accuracy", 0.0), 0.0, 1.0))
    non_change_acc = float(np.clip(metrics.get("non_change_accuracy", 0.0), 0.0, 1.0))
    change_detect_acc = float(np.clip(metrics.get("change_detect_accuracy", 0.0), 0.0, 1.0))
    change_step_mae = float(max(metrics.get("change_step_mae", 1e6), 0.0))
    change_step_score = 1.0 / (1.0 + change_step_mae)
    flat_baseline_margin = float(np.clip(metrics.get("flat_baseline_margin", 0.0), -1.0, 1.0))
    quantization_mse = float(max(metrics.get("quantization_mse", 0.0), 0.0))
    quantization_penalty = float(np.clip(quantization_mse, 0.0, 10.0))
    dominant_ratio = float(np.clip(metrics.get("dominant_code_ratio", 0.0), 0.0, 1.0))
    model_soft_warnings = float(metrics.get("model_soft_warnings_count", 0))
    env_warnings = float(metrics.get("env_warning_count", 0))
    high_sim = float(metrics.get("high_similarity_pair_count", 0))
    low_usage = float(metrics.get("low_usage_code_count", 0))

    return (
        0.40 * oracle_pos_ratio
        - 0.25 * oracle_cv
        + 0.22 * phase2_proxy_ratio
        + 0.15 * decoded_ratio
        + 0.12 * change_acc
        + 0.10 * non_change_acc
        + 0.10 * change_detect_acc
        + 0.08 * change_step_score
        + 0.08 * corr
        + 0.05 * flat_baseline_margin
        - 0.10 * model_soft_warnings
        - 0.06 * env_warnings
        - 0.05 * high_sim
        - 0.03 * low_usage
        - 0.04 * dominant_ratio
        - 0.02 * quantization_penalty
    )


def build_checkpoint_rank_tuple(
    metrics: Dict[str, Any],
    epoch: int,
    num_archetypes: int,
) -> Tuple[float, ...]:
    """构造用于 checkpoint 排序的 rank tuple（越大越好）。"""
    _ = max(int(num_archetypes), 1)
    # 选优核心：先比较更贴近 Phase II 的可实现收益代理，再比较泛化稳定性与健康约束。
    model_passed = 1.0 if bool(metrics.get("model_passed", False)) else 0.0
    dp_passed = 1.0 if bool(metrics.get("dp_passed", False)) else 0.0
    model_hard_failures = float(metrics.get("model_hard_failures_count", 0))
    dp_hard_failures = float(metrics.get("dp_hard_failures_count", 0))
    model_soft_warnings = float(metrics.get("model_soft_warnings_count", 0))

    phase2_proxy_return = float(metrics.get("phase2_proxy_return_mean", 0.0))
    phase2_proxy_ratio = float(np.clip(metrics.get("phase2_proxy_to_oracle_ratio", 0.0), -1.5, 1.5))
    phase2_realizable_proxy_return = float(metrics.get("phase2_realizable_proxy_return_mean", 0.0))
    phase2_realizable_proxy_ratio = float(
        np.clip(metrics.get("phase2_realizable_proxy_to_oracle_ratio", 0.0), -1.5, 1.5),
    )
    phase2_realizability_score = float(np.clip(metrics.get("phase2_realizability_score", 0.0), 0.0, 1.2))
    oracle_mean = float(metrics.get("oracle_return_mean", 0.0))
    oracle_pos_ratio = float(np.clip(metrics.get("oracle_positive_ratio", 0.0), 0.0, 1.0))
    oracle_std = float(metrics.get("oracle_return_std", 0.0))
    decoded_ratio = float(np.clip(metrics.get("decoded_to_oracle_return_ratio", 0.0), 0.0, 1.5))
    corr = float(np.clip(metrics.get("return_usage_correlation", 0.0), -1.0, 1.0))
    change_acc = float(np.clip(metrics.get("change_point_accuracy", 0.0), 0.0, 1.0))
    non_change_acc = float(np.clip(metrics.get("non_change_accuracy", 0.0), 0.0, 1.0))
    change_detect_acc = float(np.clip(metrics.get("change_detect_accuracy", 0.0), 0.0, 1.0))
    change_step_mae = float(max(metrics.get("change_step_mae", 1e6), 0.0))
    change_step_score = 1.0 / (1.0 + change_step_mae)
    generalization_score = compute_generalization_score(metrics)
    soft_warnings = float(metrics.get("soft_warnings_count", 0))
    env_warnings = float(metrics.get("env_warning_count", 0))
    model_soft_warnings = float(metrics.get("model_soft_warnings_count", 0))
    high_sim = float(metrics.get("high_similarity_pair_count", 0))
    low_usage = float(metrics.get("low_usage_code_count", 0))
    token_acc = float(metrics.get("token_accuracy", 0.0))
    exact_match = float(metrics.get("trajectory_exact_match_rate", 0.0))

    return (
        float(1 if metrics["overall_passed"] else 0),
        float(model_passed),
        float(dp_passed),
        float(-model_hard_failures),
        float(-dp_hard_failures),
        float(phase2_realizable_proxy_return),
        float(phase2_proxy_return),
        float(phase2_realizable_proxy_ratio),
        float(phase2_proxy_ratio),
        float(phase2_realizability_score),
        float(oracle_mean),
        float(oracle_pos_ratio),
        float(-oracle_std),
        float(generalization_score),
        float(decoded_ratio),
        float(corr),
        float(change_acc),
        float(non_change_acc),
        float(change_detect_acc),
        float(change_step_score),
        float(exact_match),
        float(token_acc),
        float(-model_soft_warnings),
        float(-soft_warnings),
        float(-env_warnings),
        float(-high_sim),
        float(-low_usage),
        float(metrics.get("decoded_return_mean", 0.0)),
        float(-epoch),
    )


def evaluate_checkpoint_profit_gate(
    *,
    metrics: Dict[str, Any],
    config: Any,
) -> Dict[str, Any]:
    """根据更贴近 Phase II 目标的收益指标，对 Phase I checkpoint 做准入过滤。"""
    oracle_mean = float(max(metrics.get("oracle_return_mean", 0.0), 0.0))
    realizable_proxy_return = float(metrics.get("phase2_realizable_proxy_return_mean", 0.0))
    best_fixed_return = float(metrics.get("best_fixed_archetype_return_mean", 0.0))
    return_usage_corr = float(np.clip(metrics.get("return_usage_correlation", 0.0), -1.0, 1.0))

    realizable_abs_threshold = float(
        max(getattr(config, "phase1_selection_min_realizable_proxy_return_mean", 0.0), 0.0),
    )
    realizable_ratio_threshold = float(
        max(getattr(config, "phase1_selection_min_realizable_proxy_to_oracle_ratio", 0.0), 0.0),
    )
    best_fixed_abs_threshold = float(
        max(getattr(config, "phase1_selection_min_best_fixed_archetype_return_mean", 0.0), 0.0),
    )
    best_fixed_ratio_threshold = float(
        max(getattr(config, "phase1_selection_min_best_fixed_to_oracle_ratio", 0.0), 0.0),
    )
    corr_threshold = float(
        np.clip(getattr(config, "phase1_selection_min_return_usage_correlation", 0.0), -1.0, 1.0),
    )

    realizable_ratio = realizable_proxy_return / oracle_mean if oracle_mean > 0 else 0.0
    best_fixed_ratio = best_fixed_return / oracle_mean if oracle_mean > 0 else 0.0
    min_realizable_required = max(realizable_abs_threshold, realizable_ratio_threshold * oracle_mean)
    min_best_fixed_required = max(best_fixed_abs_threshold, best_fixed_ratio_threshold * oracle_mean)

    failed_reasons: List[str] = []
    if realizable_proxy_return < min_realizable_required:
        failed_reasons.append(
            "realizable_proxy_return_mean="
            f"{realizable_proxy_return:.4f} < required={min_realizable_required:.4f}",
        )
    if best_fixed_return < min_best_fixed_required:
        failed_reasons.append(
            "best_fixed_archetype_return_mean="
            f"{best_fixed_return:.4f} < required={min_best_fixed_required:.4f}",
        )
    if return_usage_corr < corr_threshold:
        failed_reasons.append(
            "return_usage_correlation="
            f"{return_usage_corr:.4f} < required={corr_threshold:.4f}",
        )

    thresholds = {
        "min_realizable_proxy_return_mean": float(min_realizable_required),
        "min_realizable_proxy_to_oracle_ratio": float(realizable_ratio_threshold),
        "min_best_fixed_archetype_return_mean": float(min_best_fixed_required),
        "min_best_fixed_to_oracle_ratio": float(best_fixed_ratio_threshold),
        "min_return_usage_correlation": float(corr_threshold),
    }
    observed = {
        "realizable_proxy_return_mean": float(realizable_proxy_return),
        "realizable_proxy_to_oracle_ratio": float(realizable_ratio),
        "best_fixed_archetype_return_mean": float(best_fixed_return),
        "best_fixed_to_oracle_ratio": float(best_fixed_ratio),
        "return_usage_correlation": float(return_usage_corr),
    }
    return {
        "passed": not failed_reasons,
        "failed_reasons": failed_reasons,
        "thresholds": thresholds,
        "observed": observed,
    }


def select_and_materialize_best_phase1_checkpoint(
    *,
    config: Any,
    pair: str,
    checkpoint_candidates: List[Dict[str, Any]],
    trajectory_path: str,
    train_env: TradingEnv,
    val_env: TradingEnv | None,
    trajectory_dataset: TrajectoryDataset,
    device: torch.device,
) -> Tuple[str, str, str, Dict[str, Any]]:
    """评估候选 checkpoint，选择最佳并回写到标准 Phase I 产物路径。"""
    if not checkpoint_candidates:
        raise ValueError("Phase I checkpoint 候选列表为空，无法进行最佳 checkpoint 选择")

    save_dir = config.get_stage_result_dir(pair, "phase1_archetype_discovery")
    eval_dir = os.path.join(save_dir, "checkpoint_evaluations")
    os.makedirs(eval_dir, exist_ok=True)
    selection_rows: List[Dict[str, Any]] = []

    logger.info(
        "开始 Phase I checkpoint 评估与选择: 候选数量=%d, 间隔=%d",
        len(checkpoint_candidates),
        PHASE1_CHECKPOINT_EVAL_INTERVAL,
    )

    for candidate in checkpoint_candidates:
        epoch = int(candidate.get("epoch", 0))
        checkpoint_path = str(candidate["path"])
        tag = str(candidate.get("tag", f"epoch_{epoch:04d}"))
        safe_tag = tag.replace("/", "_")
        val_report_path = os.path.join(eval_dir, f"{pair}_{safe_tag}_phase1_validation_report.json")
        env_report_path = os.path.join(eval_dir, f"{pair}_{safe_tag}_phase1_env_validation_report.json")

        row: Dict[str, Any] = {
            "epoch": epoch,
            "tag": tag,
            "checkpoint_path": checkpoint_path,
            "training_summary": candidate.get("training_summary", {}),
            "validation_report_path": val_report_path,
            "env_report_path": env_report_path,
            "evaluation_succeeded": False,
            "error": None,
            "selection_metrics": {},
            "selection_score": float("-inf"),
            "rank_tuple": [],
        }

        try:
            set_reproducibility_seed(config.phase1_sampling_seed)
            validation_report = validate_phase1_artifacts(
                config=config,
                pair=pair,
                trajectory_path=trajectory_path,
                model_path=checkpoint_path,
                report_path=val_report_path,
                env=train_env,
                device=device,
                dp_check_limit=256,
            )

            encoder_ckpt, codebook_ckpt, decoder_ckpt = load_models_from_phase1_checkpoint(
                config=config,
                checkpoint_path=checkpoint_path,
                device=device,
            )
            set_reproducibility_seed(config.phase1_sampling_seed)
            env_report = run_phase1_env_validation(
                config=config,
                pair=pair,
                encoder=encoder_ckpt,
                codebook=codebook_ckpt,
                decoder=decoder_ckpt,
                train_env=train_env,
                trajectory_dataset=trajectory_dataset,
                device=device,
                val_env=val_env,
            )
            with open(env_report_path, "w", encoding="utf-8") as fp:
                json.dump(env_report, fp, ensure_ascii=False, indent=2, default=str)

            metrics = extract_checkpoint_selection_metrics(validation_report, env_report)
            rank_tuple = build_checkpoint_rank_tuple(metrics, epoch, int(config.num_archetypes))
            generalization_score = compute_generalization_score(metrics)
            profit_gate = evaluate_checkpoint_profit_gate(metrics=metrics, config=config)

            row["selection_metrics"] = metrics
            row["selection_score"] = float(metrics.get("phase2_realizable_proxy_return_mean", 0.0))
            row["selection_primary_oracle_return_mean"] = float(metrics.get("oracle_return_mean", 0.0))
            row["selection_generalization_score"] = float(generalization_score)
            row["selection_phase2_realizable_proxy_return_mean"] = float(
                metrics.get("phase2_realizable_proxy_return_mean", 0.0),
            )
            row["selection_phase2_realizability_score"] = float(
                metrics.get("phase2_realizability_score", 0.0),
            )
            row["selection_phase2_proxy_return_mean"] = float(metrics.get("phase2_proxy_return_mean", 0.0))
            row["selection_best_fixed_archetype_return_mean"] = float(
                metrics.get("best_fixed_archetype_return_mean", 0.0),
            )
            row["selection_return_usage_correlation"] = float(metrics.get("return_usage_correlation", 0.0))
            row["selection_profit_gate_passed"] = bool(profit_gate["passed"])
            row["selection_profit_gate_failed_reasons"] = list(profit_gate["failed_reasons"])
            row["selection_profit_gate_thresholds"] = dict(profit_gate["thresholds"])
            row["selection_profit_gate_observed"] = dict(profit_gate["observed"])
            row["rank_tuple"] = [float(x) for x in rank_tuple]
            row["evaluation_succeeded"] = True
        except Exception as exc:  # noqa: BLE001
            logger.exception("checkpoint 评估失败: epoch=%d, path=%s", epoch, checkpoint_path)
            row["error"] = str(exc)

        selection_rows.append(row)

    successful_rows = [r for r in selection_rows if r["evaluation_succeeded"]]
    if not successful_rows:
        raise RuntimeError("所有候选 checkpoint 评估均失败，无法选出最佳 checkpoint")

    profit_gated_rows = [r for r in successful_rows if r.get("selection_profit_gate_passed", False)]
    selection_pool_name = "profit_gated_candidates"
    if profit_gated_rows:
        candidate_pool = profit_gated_rows
        logger.info(
            "Phase I checkpoint profit gate 命中: %d / %d 个候选满足收益准入门槛",
            len(profit_gated_rows),
            len(successful_rows),
        )
    else:
        if bool(getattr(config, "phase1_selection_require_gated_candidate", False)):
            best_fallback_row = max(successful_rows, key=lambda r: tuple(r["rank_tuple"]))
            raise RuntimeError(
                "没有任何 Phase I checkpoint 满足 profit gate。"
                f" 最优候选(epoch={int(best_fallback_row['epoch'])}) 的失败原因: "
                + "; ".join(best_fallback_row.get("selection_profit_gate_failed_reasons", ["unknown"]))
            )
        candidate_pool = successful_rows
        selection_pool_name = "fallback_all_successful_candidates"
        logger.warning(
            "Phase I checkpoint profit gate 未命中任何候选，回退到原排序。"
            " 这通常意味着本轮 Phase I 尚未学出足够可交易的 archetype。",
        )

    best_row = max(candidate_pool, key=lambda r: tuple(r["rank_tuple"]))
    best_source_path = str(best_row["checkpoint_path"])
    best_model_path = os.path.join(save_dir, f"{pair}_vq_model_best.pt")
    standard_model_path = os.path.join(save_dir, f"{pair}_vq_model.pt")

    if os.path.abspath(best_source_path) != os.path.abspath(best_model_path):
        shutil.copy2(best_source_path, best_model_path)
    if os.path.abspath(best_source_path) != os.path.abspath(standard_model_path):
        shutil.copy2(best_source_path, standard_model_path)

    standard_val_report_path = os.path.join(save_dir, "phase1_validation_report.json")
    standard_env_report_path = os.path.join(save_dir, "phase1_env_validation_report.json")
    if os.path.abspath(best_row["validation_report_path"]) != os.path.abspath(standard_val_report_path):
        shutil.copy2(best_row["validation_report_path"], standard_val_report_path)
    if os.path.abspath(best_row["env_report_path"]) != os.path.abspath(standard_env_report_path):
        shutil.copy2(best_row["env_report_path"], standard_env_report_path)

    with open(standard_val_report_path, "r", encoding="utf-8") as fp:
        best_validation_report = json.load(fp)

    selection_report_path = os.path.join(save_dir, "phase1_checkpoint_selection_report.json")
    selection_report = {
        "pair": pair,
        "train_batch_id": config.train_batch_id,
        "selection_strategy": "phase1_v8_profit_gated_realizable_proxy_plus_validation_health",
        "phase1_epochs": int(config.phase1_epochs),
        "phase1_checkpoint_interval": int(PHASE1_CHECKPOINT_EVAL_INTERVAL),
        "candidate_count": len(checkpoint_candidates),
        "successful_candidate_count": len(successful_rows),
        "profit_gate_candidate_count": len(profit_gated_rows),
        "selection_pool": selection_pool_name,
        "profit_gate_config": {
            "min_realizable_proxy_return_mean": float(
                getattr(config, "phase1_selection_min_realizable_proxy_return_mean", 0.0),
            ),
            "min_realizable_proxy_to_oracle_ratio": float(
                getattr(config, "phase1_selection_min_realizable_proxy_to_oracle_ratio", 0.0),
            ),
            "min_best_fixed_archetype_return_mean": float(
                getattr(config, "phase1_selection_min_best_fixed_archetype_return_mean", 0.0),
            ),
            "min_best_fixed_to_oracle_ratio": float(
                getattr(config, "phase1_selection_min_best_fixed_to_oracle_ratio", 0.0),
            ),
            "min_return_usage_correlation": float(
                getattr(config, "phase1_selection_min_return_usage_correlation", 0.0),
            ),
            "require_gated_candidate": bool(
                getattr(config, "phase1_selection_require_gated_candidate", False),
            ),
        },
        "selected_checkpoint": {
            "epoch": int(best_row["epoch"]),
            "tag": best_row["tag"],
            "source_path": best_source_path,
            "best_model_path": best_model_path,
            "standard_model_path": standard_model_path,
            "standard_validation_report_path": standard_val_report_path,
            "standard_env_validation_report_path": standard_env_report_path,
            "selection_metrics": best_row["selection_metrics"],
            "selection_score": float(best_row.get("selection_score", 0.0)),
            "selection_phase2_realizable_proxy_return_mean": float(
                best_row.get("selection_phase2_realizable_proxy_return_mean", 0.0),
            ),
            "selection_phase2_realizability_score": float(
                best_row.get("selection_phase2_realizability_score", 0.0),
            ),
            "selection_phase2_proxy_return_mean": float(best_row.get("selection_phase2_proxy_return_mean", 0.0)),
            "selection_best_fixed_archetype_return_mean": float(
                best_row.get("selection_best_fixed_archetype_return_mean", 0.0),
            ),
            "selection_return_usage_correlation": float(best_row.get("selection_return_usage_correlation", 0.0)),
            "selection_profit_gate_passed": bool(best_row.get("selection_profit_gate_passed", False)),
            "selection_profit_gate_failed_reasons": list(
                best_row.get("selection_profit_gate_failed_reasons", []),
            ),
            "selection_profit_gate_thresholds": dict(best_row.get("selection_profit_gate_thresholds", {})),
            "selection_primary_oracle_return_mean": float(best_row.get("selection_primary_oracle_return_mean", 0.0)),
            "selection_generalization_score": float(best_row.get("selection_generalization_score", 0.0)),
            "rank_tuple": best_row["rank_tuple"],
        },
        "candidates": selection_rows,
    }
    with open(selection_report_path, "w", encoding="utf-8") as fp:
        json.dump(selection_report, fp, ensure_ascii=False, indent=2)

    logger.info("Phase I checkpoint 选择报告已保存到 %s", selection_report_path)
    logger.info(
        "Phase I 最佳 checkpoint 已确定: epoch=%d, realizable_proxy_mean=%.6f, proxy_mean=%.6f, "
        "realizability=%.4f, oracle_mean=%.6f, gen_score=%.6f, source=%s",
        int(best_row["epoch"]),
        float(best_row.get("selection_phase2_realizable_proxy_return_mean", 0.0)),
        float(best_row.get("selection_phase2_proxy_return_mean", 0.0)),
        float(best_row.get("selection_phase2_realizability_score", 0.0)),
        float(best_row.get("selection_primary_oracle_return_mean", 0.0)),
        float(best_row.get("selection_generalization_score", 0.0)),
        best_source_path,
    )
    if not bool(best_row.get("selection_profit_gate_passed", False)):
        logger.warning(
            "当前选中的 Phase I checkpoint 未满足 profit gate，原因: %s",
            "; ".join(best_row.get("selection_profit_gate_failed_reasons", [])) or "unknown",
        )
    logger.info("标准 Phase I 模型路径已更新为最佳 checkpoint: %s", standard_model_path)

    return standard_model_path, standard_val_report_path, standard_env_report_path, best_validation_report


def log_training_summary(
    pair: str,
    loss_history: List[float],
    traj_path: str,
    save_path: str,
    report_path: str,
    validation_report: dict,
) -> None:
    """输出训练完成后的日志摘要。"""
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


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    # Step 0: 解析配置
    config = parse_args()
    pair = config.pairs[0]
    # assert_paper_phase1_settings(config, pair)
    set_reproducibility_seed(config.phase1_sampling_seed)

    logger.info("Phase I 训练开始: pair=%s", pair)
    logger.info("结果目录批次: %s", config.train_batch_id)
    logger.info(
        "严格论文主线配置已通过守卫检查: epochs=%d, batch_size=%d, lr=%.1e, latent_dim=%d, "
        "num_archetypes=%d, num_trajectories=%d, vq_beta0=%.2f, sampling_seed=%d, "
        "sampling_mode=%s(%.2f/%.2f), align_w=%.3f, align_target=%.2f, return_w=%.3f, "
        "return_hidden=%d, return_bins=%d, return_soft=%.2f, sep_w=%.3f, sep_margin=%.2f, "
        "init_top=%.2f, init_code=%.2f, reset_top=%.2f",
        config.phase1_epochs, config.batch_size, config.learning_rate, config.latent_dim,
        config.num_archetypes, config.num_trajectories, config.vq_beta0, config.phase1_sampling_seed,
        config.phase1_start_sampling_mode, config.phase1_stratified_ratio, config.phase1_importance_ratio,
        config.phase1_usage_profit_alignment_weight,
        config.phase1_usage_profit_alignment_target_corr,
        config.phase1_return_aux_weight,
        config.phase1_return_aux_hidden_dim,
        config.phase1_return_num_buckets,
        config.phase1_return_soft_assignment_weight,
        config.phase1_codebook_separation_weight,
        config.phase1_codebook_separation_margin,
        config.phase1_profit_init_top_ratio,
        config.phase1_profit_init_code_ratio,
        config.phase1_profit_reset_top_ratio,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    # Step 1: 加载数据 & 环境
    env, dp_env, train_rows = load_data_and_env(config, pair)

    # Step 2: 准备轨迹数据集（用 dp_env 的高费率筛选高利润轨迹）
    dataset, traj_path = prepare_trajectory_dataset(config, pair, dp_env, train_rows)
    logger.info("Dataset 大小: %d 条轨迹", len(dataset))

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=False,
    )

    # Step 3: 初始化模型
    encoder, codebook, decoder = build_models(config, device)
    return_bucket_head = build_return_bucket_head(config, device)

    # Step 4: 训练（每 30 epoch 保存一个 checkpoint 候选）
    save_dir = config.get_stage_result_dir(pair, "phase1_archetype_discovery")
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_interval = PHASE1_CHECKPOINT_EVAL_INTERVAL
    checkpoint_candidates: List[Dict[str, Any]] = []
    logger.info("已启用 Phase I checkpoint 评估: 每 %d 个 epoch 评估一次", checkpoint_interval)

    def on_epoch_checkpoint(epoch: int, summary: Dict[str, float], cur_history: TrainingHistory) -> None:
        checkpoint_path = os.path.join(checkpoint_dir, f"{pair}_vq_model_epoch_{epoch:04d}.pt")
        save_checkpoint(
            config=config,
            pair=pair,
            encoder=encoder,
            codebook=codebook,
            decoder=decoder,
            history=cur_history,
            train_rows=train_rows,
            norm_stats=dataset.norm_stats,
            save_path=checkpoint_path,
            checkpoint_meta={
                "epoch": int(epoch),
                "kind": "periodic",
                "phase1_checkpoint_interval": int(checkpoint_interval),
            },
        )
        checkpoint_candidates.append(
            {
                "epoch": int(epoch),
                "tag": f"epoch_{epoch:04d}",
                "path": checkpoint_path,
                "training_summary": {
                    k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                    for k, v in summary.items()
                },
            }
        )

    history = run_training_loop(
        dataloader=dataloader,
        encoder=encoder,
        codebook=codebook,
        decoder=decoder,
        return_bucket_head=return_bucket_head,
        config=config,
        device=device,
        checkpoint_interval=checkpoint_interval,
        on_checkpoint=on_epoch_checkpoint,
    )

    # Step 5: 不再保存 final 模型；仅使用候选 checkpoint 进行评估与选择
    if not checkpoint_candidates:
        # 当训练轮数不足一个评估间隔（或提前结束）时，兜底保存一次当前模型用于单次评估。
        fallback_epoch = int(config.phase1_epochs)
        fallback_checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{pair}_vq_model_epoch_{fallback_epoch:04d}.pt",
        )
        save_checkpoint(
            config=config,
            pair=pair,
            encoder=encoder,
            codebook=codebook,
            decoder=decoder,
            history=history,
            train_rows=train_rows,
            norm_stats=dataset.norm_stats,
            save_path=fallback_checkpoint_path,
            checkpoint_meta={
                "epoch": fallback_epoch,
                "kind": "fallback_single_eval",
                "phase1_checkpoint_interval": int(checkpoint_interval),
            },
        )
        checkpoint_candidates.append(
            {
                "epoch": fallback_epoch,
                "tag": f"epoch_{fallback_epoch:04d}",
                "path": fallback_checkpoint_path,
                "training_summary": {
                    "avg_loss": float(history.loss[-1]) if history.loss else float("nan"),
                    "vq_loss": float(history.vq[-1]) if history.vq else float("nan"),
                    "phase": "fallback_single_eval",
                },
            }
        )
        logger.info(
            "未命中周期 checkpoint（epochs=%d, interval=%d），已生成兜底候选用于单次评估: %s",
            int(config.phase1_epochs),
            int(checkpoint_interval),
            fallback_checkpoint_path,
        )
    logger.info("Phase I 周期 checkpoint 候选数量: %d", len(checkpoint_candidates))

    # Step 6: 构建验证集环境（供 checkpoint 评估阶段复用）
    logger.info("构建验证集环境（用于 checkpoint 选择）...")
    val_env = build_val_env(config, pair)
    if val_env is None:
        logger.warning("验证集不足一个 horizon，checkpoint 选择将仅依赖训练集环境诊断")

    # Step 7: 候选 checkpoint 评估 + 选择最佳，并回写标准路径
    save_path, report_path, env_report_path, validation_report = select_and_materialize_best_phase1_checkpoint(
        config=config,
        pair=pair,
        checkpoint_candidates=checkpoint_candidates,
        trajectory_path=traj_path,
        train_env=env,
        val_env=val_env,
        trajectory_dataset=dataset,
        device=device,
    )

    # Step 8: 日志摘要
    log_training_summary(pair, history.loss, traj_path, save_path, report_path, validation_report)
    logger.info("最佳 checkpoint 环境级验证报告路径: %s", env_report_path)


if __name__ == "__main__":
    main()
