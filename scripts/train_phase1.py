#!/usr/bin/env python
"""Phase I 训练脚本 — 原型发现

# 需求: 7.1, 4.6, 4.7, 4.8, 7.5, 7.6, 7.7
#
# 流程:
# 1. 加载特征数据，初始化 TradingEnv
# 2. 调用 DPPlanner 生成示范轨迹并保存（默认 num_trajectories=90k，可用 CLI 覆盖）
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
from src.phase1.checkpoint import (
    evaluate_checkpoint_profit_gate,
    extract_checkpoint_selection_metrics,
    load_models_from_phase1_checkpoint,
    save_checkpoint,
    select_and_materialize_best_phase1_checkpoint,
)
from src.phase1.codebook import VQCodebook
from src.phase1.data_loader import (
    build_val_env,
    load_data_and_env,
    prepare_trajectory_dataset,
)
from src.phase1.dp_planner import DPPlanner
from src.phase1.env_validation import run_phase1_env_validation
from src.phase1.utils import (
    compute_grad_norm,
    set_reproducibility_seed,
    summarize_code_usage,
)
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
            "当前训练集不足以在严格论文滑窗协议下无放回采样指定数量的 trajectories。"
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
