"""Phase I 检查点管理模块

本模块提供 Phase I 训练中的检查点保存、加载和选择功能。

Functions:
    save_checkpoint: 保存模型检查点
    load_models_from_phase1_checkpoint: 从检查点加载模型
    extract_checkpoint_selection_metrics: 提取检查点选择指标
    compute_phase2_realizability_score: 计算 Phase 2 可实现性分数
    compute_generalization_score: 计算泛化分数
    build_checkpoint_rank_tuple: 构建检查点排名元组
    evaluate_checkpoint_profit_gate: 评估检查点收益门槛
    select_and_materialize_best_phase1_checkpoint: 选择最佳检查点
"""

import json
import os
import shutil
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.data.dataset import TrajectoryDataset
from src.env.trading_env import TradingEnv
from src.phase1.codebook import VQCodebook
from src.phase1.env_validation import run_phase1_env_validation
from src.phase1.utils import set_reproducibility_seed
from src.phase1.validation import validate_phase1_artifacts
from src.phase1.vq_decoder import VQDecoder
from src.phase1.vq_encoder import VQEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Phase I checkpoint 评估间隔：每训练 30 个 epoch 评估一次
PHASE1_CHECKPOINT_EVAL_INTERVAL = 30

# 论文参考训练数据规模
PAPER_PHASE1_REFERENCE_TRAIN_ROWS = 1_400_000


def save_checkpoint(
    *,
    config: Any,
    pair: str,
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
    history: Any,  # TrainingHistory
    train_rows: int,
    norm_stats: dict | None = None,
    save_path: str | None = None,
    checkpoint_meta: Dict[str, Any] | None = None,
) -> str:
    """保存模型 checkpoint 到 result/phase1_archetype_discovery/，返回保存路径。

    Args:
        config: 配置对象
        pair: 交易对
        encoder: VQ Encoder 模型
        codebook: VQ Codebook 模型
        decoder: VQ Decoder 模型
        history: 训练历史对象
        train_rows: 训练数据行数
        norm_stats: 归一化统计信息
        save_path: 保存路径（可选）
        checkpoint_meta: 检查点元数据

    Returns:
        保存的文件路径
    """
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
            "state_dim": encoder.state_dim,
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


def load_models_from_phase1_checkpoint(
    *,
    config: Any,
    pair: str,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[VQEncoder, VQCodebook, VQDecoder]:
    """从指定 checkpoint 还原 encoder/codebook/decoder。

    Args:
        config: 配置对象
        checkpoint_path: 检查点文件路径
        device: 计算设备

    Returns:
        (encoder, codebook, decoder) 模型元组
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dim = config.get_state_dim(pair)
    encoder = VQEncoder(
        state_dim=state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.lstm_hidden_dim,
        latent_dim=config.latent_dim,
    ).to(device)
    codebook = VQCodebook(
        num_codes=config.num_archetypes,
        code_dim=config.latent_dim,
    ).to(device)
    decoder = VQDecoder(
        state_dim=state_dim,
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
                pair=pair,
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
                "selection_pool": "profit_gate_failed_no_candidate_passed",
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
                    "require_gated_candidate": True,
                },
                "selection_error": {
                    "type": "profit_gate_no_candidate_passed",
                    "best_fallback_epoch": int(best_fallback_row.get("epoch", 0)),
                    "best_fallback_failed_reasons": list(
                        best_fallback_row.get("selection_profit_gate_failed_reasons", []),
                    ),
                },
                "selected_checkpoint": {
                    "epoch": int(best_fallback_row.get("epoch", 0)),
                    "tag": str(best_fallback_row.get("tag", "")),
                    "source_path": str(best_fallback_row.get("checkpoint_path", "")),
                    "validation_report_path": str(best_fallback_row.get("validation_report_path", "")),
                    "env_report_path": str(best_fallback_row.get("env_report_path", "")),
                    "selection_metrics": best_fallback_row.get("selection_metrics", {}),
                    "selection_score": float(best_fallback_row.get("selection_score", 0.0)),
                    "selection_phase2_realizable_proxy_return_mean": float(
                        best_fallback_row.get("selection_phase2_realizable_proxy_return_mean", 0.0),
                    ),
                    "selection_phase2_realizability_score": float(
                        best_fallback_row.get("selection_phase2_realizability_score", 0.0),
                    ),
                    "selection_phase2_proxy_return_mean": float(
                        best_fallback_row.get("selection_phase2_proxy_return_mean", 0.0),
                    ),
                    "selection_best_fixed_archetype_return_mean": float(
                        best_fallback_row.get("selection_best_fixed_archetype_return_mean", 0.0),
                    ),
                    "selection_return_usage_correlation": float(
                        best_fallback_row.get("selection_return_usage_correlation", 0.0),
                    ),
                    "selection_profit_gate_passed": bool(
                        best_fallback_row.get("selection_profit_gate_passed", False),
                    ),
                    "selection_profit_gate_failed_reasons": list(
                        best_fallback_row.get("selection_profit_gate_failed_reasons", []),
                    ),
                    "selection_profit_gate_thresholds": dict(
                        best_fallback_row.get("selection_profit_gate_thresholds", {}),
                    ),
                    "selection_primary_oracle_return_mean": float(
                        best_fallback_row.get("selection_primary_oracle_return_mean", 0.0),
                    ),
                    "selection_generalization_score": float(
                        best_fallback_row.get("selection_generalization_score", 0.0),
                    ),
                    "rank_tuple": best_fallback_row.get("rank_tuple", []),
                    "materialized": False,
                },
                "candidates": selection_rows,
            }
            with open(selection_report_path, "w", encoding="utf-8") as fp:
                json.dump(selection_report, fp, ensure_ascii=False, indent=2)
            logger.error(
                "Phase I checkpoint profit gate 未命中任何候选，已写入选择报告: %s",
                selection_report_path,
            )
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
