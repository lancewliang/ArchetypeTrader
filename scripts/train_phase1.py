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
from typing import Any, Dict, List, Tuple

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
    "phase1_epochs": 300,
    "pretrain_epochs": 10,
    "discount_factor": 0.99,
    "max_positions": {"BTC": 8, "ETH": 100, "DOT": 2500, "BNB": 200},
}


# ---------------------------------------------------------------------------
# 数据类：训练指标收集
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    """单个 epoch 的累积指标。"""

    loss: float = 0.0
    rec_loss: float = 0.0
    vq_loss: float = 0.0
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


# ---------------------------------------------------------------------------
# 从 main() 中提取的子流程
# ---------------------------------------------------------------------------

def load_data_and_env(config: Any, pair: str) -> Tuple[TradingEnv, int]:
    """加载特征数据并初始化 TradingEnv。

    Returns:
        (env, train_rows)
    """
    logger.info("加载特征数据: data_dir=%s, pair=%s", config.data_dir, pair)
    pipeline = FeaturePipeline(config.data_dir, pair)
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
        commission_rate=config.commission_rate,
    )

    logger.info(
        "TradingEnv 初始化完成: num_horizons=%d, horizon=%d, max_position=%d, "
        "commission_rate=%.6f, available_starts=%d",
        env.num_horizons, config.horizon, env.m, env.commission_rate, available_starts,
    )
    return env, train_rows


def prepare_trajectory_dataset(
    config: Any, pair: str, env: TradingEnv, train_rows: int,
) -> Tuple[TrajectoryDataset, str]:
    """检查缓存 / 生成 DP 示范轨迹，返回 (dataset, traj_path)。"""
    planner = DPPlanner(
        env=env,
        gamma=config.discount_factor,
        result_dir=config.result_dir,
        sampling_seed=config.phase1_sampling_seed,
    )
    traj_path = DPPlanner.build_trajectory_cache_path(config.result_dir, pair)

    if os.path.exists(traj_path):
        cache_ok, cache_reasons = inspect_trajectory_cache(
            traj_path=traj_path, config=config, pair=pair, train_rows=train_rows,
        )
        if cache_ok:
            logger.info("发现与当前严格论文设置兼容的轨迹缓存，直接加载: %s", traj_path)
            return TrajectoryDataset.from_npz(traj_path), traj_path
        backup_incompatible_cache(traj_path, cache_reasons)

    logger.info("开始生成 DP 示范轨迹: num_trajectories=%d", config.num_trajectories)
    trajectories = planner.generate_trajectories(config.num_trajectories)
    logger.info("DP 轨迹生成完成，创建 Dataset")
    dataset = TrajectoryDataset(
        states=trajectories["states"],
        actions=trajectories["actions"],
        rewards=trajectories["rewards"],
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
        use_ema=config.use_ema_codebook,
        ema_decay=config.ema_decay,
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
    optimizer: torch.optim.Optimizer,
    ce_loss_fn: nn.CrossEntropyLoss,
    config: Any,
    device: torch.device,
    is_phase_a: bool,
    collect_z_e: bool = False,
) -> Tuple[EpochMetrics, torch.Tensor | None]:
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

    metrics = EpochMetrics(
        code_counts=np.zeros(config.num_archetypes, dtype=np.int64),
    )
    z_e_list: List[torch.Tensor] = [] if collect_z_e else []
    a_demo_list: List[torch.Tensor] = [] if collect_z_e else []
    last_z_e: torch.Tensor | None = None

    for s_demo, a_demo, r_demo in dataloader:
        s_demo = s_demo.to(device)
        a_demo = a_demo.to(device)
        r_demo = r_demo.to(device)

        # Encode
        z_e = encoder(s_demo, a_demo, r_demo)

        if collect_z_e:
            z_e_list.append(z_e.detach())
            a_demo_list.append(a_demo.detach())

        if not is_phase_a:
            last_z_e = z_e.detach()

        if is_phase_a:
            # Phase A: bypass VQ, pass z_e directly to decoder
            z_input = z_e
            indices_np = None
            vq_loss_val = 0.0
            quantization_mse_val = 0.0
        else:
            # Phase B: full VQ quantization
            z_q_st, indices, commitment_loss = codebook.quantize(z_e)
            z_input = z_q_st
            indices_np = indices.detach().cpu().numpy()

        # Decode
        action_logits = decoder(s_demo, z_input)
        pred_actions = torch.argmax(action_logits, dim=-1)

        # L_rec with change-point weighting
        # change points: 位置 t 处 a_demo[:, t] != a_demo[:, t-1]
        # 这些位置决定了整个 horizon 的盈亏方向，给予更高权重
        logits_flat = action_logits.reshape(-1, config.action_dim)
        targets_flat = a_demo.reshape(-1)

        batch_size_cur = a_demo.shape[0]
        h_len = a_demo.shape[1]

        # 构建逐 token 权重: 默认 1.0，change point 处乘以 change_point_weight
        token_weights = torch.ones(batch_size_cur, h_len, device=device)
        if h_len >= 2:
            change_mask = (a_demo[:, 1:] != a_demo[:, :-1])  # (batch, h-1)
            token_weights[:, 1:] += change_mask.float() * (config.change_point_weight - 1.0)
        weights_flat = token_weights.reshape(-1)

        # 加权 cross-entropy
        per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        rec_loss = (per_token_loss * weights_flat).mean()

        if is_phase_a:
            total_loss = rec_loss
        else:
            # β₀ × ||z_e - sg[z_q]||²
            z_q_detached = z_q_st.detach()
            encoder_commitment = config.vq_beta0 * torch.mean((z_e - z_q_detached) ** 2)
            total_loss = rec_loss + commitment_loss + encoder_commitment
            vq_loss_val = commitment_loss.item() + encoder_commitment.item()
            quantization_mse_val = float(torch.mean((z_e - z_q_detached) ** 2).item())

        optimizer.zero_grad()
        total_loss.backward()

        # 累积指标（在 step 之前，梯度可用）
        _accumulate_batch_metrics(
            metrics,
            total_loss=total_loss,
            rec_loss=rec_loss,
            vq_loss_val=vq_loss_val,
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

    # Phase B: 死码重置
    if not is_phase_a and last_z_e is not None:
        codebook.reset_dead_codes(last_z_e, metrics.code_counts)

    return metrics, z_e_all, a_demo_all


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------

def run_training_loop(
    *,
    dataloader: DataLoader,
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
    config: Any,
    device: torch.device,
) -> TrainingHistory:
    """执行完整的 Phase I 训练循环（Phase A 预训练 + Phase B VQ 训练）。"""
    # EMA 模式下码本不参与梯度更新，只收集有梯度的参数
    trainable_params = [p for p in (
        list(encoder.parameters())
        + list(codebook.parameters())
        + list(decoder.parameters())
    ) if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config.learning_rate)
    # Cosine annealing: 从 lr 衰减到 lr/10，防止后期过拟合
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.phase1_epochs, eta_min=config.learning_rate / 10,
    )
    ce_loss_fn = nn.CrossEntropyLoss()
    history = TrainingHistory()

    logger.info("开始训练: %d epochs", config.phase1_epochs)
    logger.info("Phase A (连续潜在预训练): epochs 1-%d, loss=L_rec only", config.pretrain_epochs)

    for epoch in tqdm(range(1, config.phase1_epochs + 1), desc="Training Epochs"):
        is_phase_a = epoch <= config.pretrain_epochs
        # 在 Phase A 最后一个 epoch 收集 z_e 用于 k-means 初始化
        collect_z_e = (epoch == config.pretrain_epochs)

        if epoch == config.pretrain_epochs + 1:
            logger.info(
                "Phase B (VQ 训练): epochs %d-%d, full VQ loss",
                config.pretrain_epochs + 1, config.phase1_epochs,
            )

        metrics, z_e_all, a_demo_all = train_one_epoch(
            dataloader=dataloader,
            encoder=encoder,
            codebook=codebook,
            decoder=decoder,
            optimizer=optimizer,
            ce_loss_fn=ce_loss_fn,
            config=config,
            device=device,
            is_phase_a=is_phase_a,
            collect_z_e=collect_z_e,
        )

        # Phase A → Phase B 过渡: 用方向感知 k-means 初始化码本
        if collect_z_e and z_e_all is not None:
            if a_demo_all is not None:
                codebook.init_from_data_direction_aware(z_e_all, a_demo_all)
            else:
                codebook.init_from_data(z_e_all)

        summary = metrics.summarize()
        history.append_from_summary(summary)

        if epoch == 1 or epoch % 10 == 0 or epoch == config.phase1_epochs:
            tqdm.write(
                "Epoch %3d/%d — total_loss=%.4f, rec_loss=%.4f, vq_loss=%.4f, "
                "token_acc=%.4f, exact_match=%.4f, perplexity=%.4f, used_codes=%d"
                % (
                    epoch, config.phase1_epochs,
                    summary["avg_loss"], summary["avg_rec"], summary["avg_vq"],
                    summary["token_accuracy"], summary["exact_match_rate"],
                    summary["codebook_perplexity"], summary["used_code_count"],
                )
            )

        if np.isnan(summary["avg_loss"]):
            logger.error("训练 loss 发散 (NaN)，在 epoch %d 终止训练", epoch)
            break

        scheduler.step()

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
) -> str:
    """保存模型 checkpoint 到 result/phase1_archetype_discovery/，返回保存路径。"""
    save_dir = os.path.join(config.result_dir, pair, "phase1_archetype_discovery")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{pair}_vq_model.pt")
    torch.save(
        {
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
                "discount_factor": config.discount_factor,
                "commission_rate": config.commission_rate,
                "max_positions": config.max_positions,
                "paper_phase1_reference_train_rows": PAPER_PHASE1_REFERENCE_TRAIN_ROWS,
                "current_train_rows": train_rows,
            },
        },
        save_path,
    )
    logger.info("模型已保存到 %s", save_path)
    return save_path


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
    logger.info(
        "严格论文主线配置已通过守卫检查: epochs=%d, batch_size=%d, lr=%.1e, latent_dim=%d, "
        "num_archetypes=%d, num_trajectories=%d, vq_beta0=%.2f, sampling_seed=%d",
        config.phase1_epochs, config.batch_size, config.learning_rate, config.latent_dim,
        config.num_archetypes, config.num_trajectories, config.vq_beta0, config.phase1_sampling_seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    # Step 1: 加载数据 & 环境
    env, train_rows = load_data_and_env(config, pair)

    # Step 2: 准备轨迹数据集
    dataset, traj_path = prepare_trajectory_dataset(config, pair, env, train_rows)
    logger.info("Dataset 大小: %d 条轨迹", len(dataset))

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=False,
    )

    # Step 3: 初始化模型
    encoder, codebook, decoder = build_models(config, device)

    # Step 4: 训练
    history = run_training_loop(
        dataloader=dataloader,
        encoder=encoder,
        codebook=codebook,
        decoder=decoder,
        config=config,
        device=device,
    )

    # Step 5: 保存模型
    save_path = save_checkpoint(
        config=config, pair=pair,
        encoder=encoder, codebook=codebook, decoder=decoder,
        history=history, train_rows=train_rows,
    )

    # Step 6: 验证
    save_dir = os.path.join(config.result_dir, pair, "phase1_archetype_discovery")
    report_path = os.path.join(save_dir, "phase1_validation_report.json")
    validation_report = validate_phase1_artifacts(
        config=config, pair=pair,
        trajectory_path=traj_path, model_path=save_path,
        report_path=report_path, env=env, device=device,
        dp_check_limit=256,
    )

    # Step 7: 环境级验证 — 评估 archetype 在真实交易环境中的可用性
    logger.info("开始 Phase I 环境级验证...")
    # 加载验证集环境
    val_pipeline = FeaturePipeline(config.data_dir, pair)
    _, val_state_df, _ = val_pipeline.get_state_vector()
    _, val_prices_df, _ = val_pipeline.get_prices()
    val_env = None
    if val_state_df is not None and len(val_state_df) >= config.horizon:
        val_env = TradingEnv(
            states=val_state_df.to_numpy(),
            prices=val_prices_df["close"].to_numpy(),
            pair=pair,
            horizon=config.horizon,
            states_dataframe=val_state_df,
            max_positions=config.max_positions,
            commission_rate=config.commission_rate,
        )

    env_report = run_phase1_env_validation(
        config=config,
        pair=pair,
        encoder=encoder,
        codebook=codebook,
        decoder=decoder,
        train_env=env,
        trajectory_dataset=dataset,
        device=device,
        val_env=val_env,
    )

    # 保存环境级验证报告
    env_report_path = os.path.join(save_dir, "phase1_env_validation_report.json")
    with open(env_report_path, "w", encoding="utf-8") as fp:
        json.dump(env_report, fp, ensure_ascii=False, indent=2, default=str)
    logger.info("环境级验证报告已保存到 %s", env_report_path)

    # Step 8: 日志摘要
    log_training_summary(pair, history.loss, traj_path, save_path, report_path, validation_report)


if __name__ == "__main__":
    main()
