"""Phase I 验证模块 — 验证 DP 轨迹与 VQ 模型的一致性。

该模块的目标是为 Phase I 提供可重复、可落盘的验证报告，覆盖两大部分：
1. dp_trajectories 的结构完整性、single-trade 约束、reward 回放与 DP 策略一致性。
2. phase1_model 的 checkpoint 完整性、损失分解、动作重建质量、codebook 使用情况与几何健康度。

所有关键指标都会汇总到 phase1_validation_report.json，便于后续 debug。
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import TrajectoryDataset
from src.data.feature_pipeline import FeaturePipeline
from src.env.trading_env import TradingEnv
from src.phase1.codebook import VQCodebook
from src.phase1.dp_planner import DPPlanner
from src.phase1.vq_decoder import VQDecoder
from src.phase1.vq_encoder import VQEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _to_serializable(value: Any) -> Any:
    """将 numpy / torch 对象递归转换为 JSON 可序列化对象。"""
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _safe_mean(array: np.ndarray) -> float:
    """计算均值；空数组时返回 0。"""
    if array.size == 0:
        return 0.0
    return float(np.mean(array))


def _safe_std(array: np.ndarray) -> float:
    """计算标准差；空数组时返回 0。"""
    if array.size == 0:
        return 0.0
    return float(np.std(array))


def _safe_min(array: np.ndarray) -> float:
    """计算最小值；空数组时返回 0。"""
    if array.size == 0:
        return 0.0
    return float(np.min(array))


def _safe_max(array: np.ndarray) -> float:
    """计算最大值；空数组时返回 0。"""
    if array.size == 0:
        return 0.0
    return float(np.max(array))


def _build_check_indices(total: int, max_checks: int) -> np.ndarray:
    """构造用于重放 / 重规划验证的子样本索引。"""
    if total <= 0:
        return np.empty(0, dtype=np.int64)
    if total <= max_checks:
        return np.arange(total, dtype=np.int64)
    return np.unique(np.linspace(0, total - 1, num=max_checks, dtype=np.int64))


def _histogram_from_int_array(values: np.ndarray) -> Dict[str, int]:
    """将整数数组统计成 JSON 友好的直方图。"""
    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def _compute_entropy_from_counts(counts: np.ndarray) -> float:
    """根据计数数组计算香农熵。"""
    total = int(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = counts.astype(np.float64) / float(total)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def _compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """计算参数梯度的全局 L2 范数。"""
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().data.norm(2).item()
        total += grad_norm * grad_norm
    return float(total ** 0.5)


def replay_rewards_from_actions(
    env: TradingEnv,
    start_index: int,
    actions: np.ndarray,
) -> np.ndarray:
    """基于环境定义重放动作序列，重算 reward。

    该方法用于验证 dp_trajectories 中保存的 reward 是否与环境公式完全一致。

    Args:
        env: 交易环境实例
        start_index: 该轨迹在原始训练集中的起点
        actions: 动作序列 shape (h,)

    Returns:
        重算得到的 reward 序列，shape (h,)
    """
    horizon = len(actions)
    rewards = np.zeros(horizon, dtype=np.float64)
    current_position = 0

    for local_t, action in enumerate(actions):
        global_t = start_index + local_t
        p_t = env.prices[global_t]
        p_next = env.prices[global_t + 1] if (global_t + 1) < len(env.prices) else env.prices[global_t]
        state_row = env.states_dataframe.row(global_t, named=True) if env.states_dataframe is not None else None
        execution_cost = env.compute_execution_cost(int(action), current_position, p_t, state_row)
        next_position = TradingEnv.POSITION_MAP[int(action)] * env.m
        rewards[local_t] = next_position * (p_next - p_t) - execution_cost
        current_position = next_position

    return rewards


def compute_bellman_residuals(
    planner: DPPlanner,
    env: TradingEnv,
    states: Any,
    prices: np.ndarray,
    debug_info: Dict[str, Any],
) -> Dict[str, float]:
    """验证 DP 表是否满足 Bellman 一致性。"""
    V = debug_info["V"]
    Pi = debug_info["Pi"]
    N = len(states)
    residuals: List[float] = []

    for t in range(N):
        p_t = prices[t]
        p_next = prices[t + 1] if (t + 1) < len(prices) else prices[t]
        state_row = states.row(t, named=True)
        for a_prev in range(planner.NUM_ACTIONS):
            prev_position = TradingEnv.POSITION_MAP[a_prev] * env.m
            for c in range(planner.C):
                a_next = int(Pi[t, a_prev, c])
                c_next = planner._compute_next_constraint(c, a_prev, a_next)
                if c_next < 0:
                    continue
                next_position = TradingEnv.POSITION_MAP[a_next] * env.m
                execution_cost = env.compute_execution_cost(a_next, prev_position, p_t, state_row)
                reward = next_position * (p_next - p_t) - execution_cost
                rhs = reward + planner.gamma * V[t + 1, a_next, c_next]
                residuals.append(abs(float(V[t, a_prev, c]) - float(rhs)))

    residual_array = np.asarray(residuals, dtype=np.float64)
    return {
        "bellman_residual_mean": _safe_mean(residual_array),
        "bellman_residual_max": _safe_max(residual_array),
    }


def validate_single_trade_constraint(actions: np.ndarray) -> Dict[str, Any]:
    """验证 single-trade / single-change 约束是否成立。"""
    if actions.ndim != 2:
        raise ValueError(f"actions 应为 2D (N, h)，实际为 {actions.ndim}D")

    if actions.shape[1] >= 2:
        change_counts = np.sum(actions[:, 1:] != actions[:, :-1], axis=1).astype(np.int64)
        last_step_copy_mask = actions[:, -1] == actions[:, -2]
    else:
        change_counts = np.zeros(actions.shape[0], dtype=np.int64)
        last_step_copy_mask = np.ones(actions.shape[0], dtype=bool)

    violations = change_counts > 1
    return {
        "num_trajectories": int(actions.shape[0]),
        "action_change_histogram": _histogram_from_int_array(change_counts),
        "num_single_trade_violations": int(np.sum(violations)),
        "single_trade_violation_ratio": float(np.mean(violations)) if violations.size else 0.0,
        "last_step_copy_violation_count": int(np.sum(~last_step_copy_mask)),
    }


def validate_dp_trajectories(
    config: Any,
    pair: str,
    env: TradingEnv,
    trajectory_path: str,
    dp_check_limit: int = 256,
) -> Dict[str, Any]:
    """验证 dp_trajectories 文件的完整性、约束和数学一致性。"""
    report: Dict[str, Any] = {
        "file_integrity": {},
        "sampling": {},
        "single_trade_constraint": {},
        "reward_replay": {},
        "bellman_consistency": {},
        "trajectory_stats": {},
        "bad_case_examples": [],
        "hard_failures": [],
        "soft_warnings": [],
    }

    data = np.load(trajectory_path, allow_pickle=False)
    available_keys = set(data.files)
    required_keys = {"states", "actions", "rewards"}
    missing_keys = sorted(required_keys - available_keys)
    if missing_keys:
        report["hard_failures"].append(f"dp_trajectories 缺少必要键: {missing_keys}")
        return report

    states = data["states"]
    actions = data["actions"]
    rewards = data["rewards"]

    sampled_start_indices = data["sampled_start_indices"] if "sampled_start_indices" in data.files else None
    sampling_seed = int(data["sampling_seed"]) if "sampling_seed" in data.files else None
    replace = bool(data["replace"]) if "replace" in data.files else None
    num_available_starts = int(data["num_available_starts"]) if "num_available_starts" in data.files else None
    num_sampled_trajectories = int(data["num_sampled_trajectories"]) if "num_sampled_trajectories" in data.files else int(states.shape[0])

    file_integrity = {
        "available_keys": sorted(available_keys),
        "states_shape": list(states.shape),
        "actions_shape": list(actions.shape),
        "rewards_shape": list(rewards.shape),
        "states_has_nan": bool(np.isnan(states).any()),
        "states_has_inf": bool(np.isinf(states).any()),
        "rewards_has_nan": bool(np.isnan(rewards).any()),
        "rewards_has_inf": bool(np.isinf(rewards).any()),
        "actions_unique_values": sorted(int(x) for x in np.unique(actions)),
        "expected_horizon": int(config.horizon),
        "expected_state_dim": int(config.state_dim),
        "expected_num_trajectories": int(config.num_trajectories),
        "actual_num_trajectories": int(states.shape[0]),
    }
    report["file_integrity"] = file_integrity

    if states.ndim != 3:
        report["hard_failures"].append(f"states 维度错误: 期望 3D，实际 {states.ndim}D")
    if actions.ndim != 2:
        report["hard_failures"].append(f"actions 维度错误: 期望 2D，实际 {actions.ndim}D")
    if rewards.ndim != 2:
        report["hard_failures"].append(f"rewards 维度错误: 期望 2D，实际 {rewards.ndim}D")
    if states.ndim == 3 and states.shape[1] != config.horizon:
        report["hard_failures"].append(
            f"轨迹 horizon 与 config 不一致: actions h={states.shape[1]}, expected={config.horizon}"
        )
    if states.ndim == 3 and states.shape[2] != config.state_dim:
        report["hard_failures"].append(
            f"state_dim 与 config 不一致: states dim={states.shape[2]}, expected={config.state_dim}"
        )
    if states.shape[0] != actions.shape[0] or states.shape[0] != rewards.shape[0]:
        report["hard_failures"].append("states/actions/rewards 样本数不一致")
    if np.isnan(states).any() or np.isinf(states).any() or np.isnan(rewards).any() or np.isinf(rewards).any():
        report["hard_failures"].append("states 或 rewards 中存在 NaN/Inf")
    if not np.isin(actions, [0, 1, 2]).all():
        report["hard_failures"].append("actions 中存在非法动作值，合法集合应为 {0,1,2}")
    if int(states.shape[0]) != int(num_sampled_trajectories):
        report["hard_failures"].append(
            f"num_sampled_trajectories 与实际轨迹数不一致: meta={num_sampled_trajectories}, actual={states.shape[0]}"
        )

    # 采样元数据
    if sampled_start_indices is not None and sampled_start_indices.size > 0:
        unique_starts = np.unique(sampled_start_indices)
        duplicate_ratio = 1.0 - float(len(unique_starts)) / float(len(sampled_start_indices))
        report["sampling"] = {
            "has_sampling_metadata": True,
            "sampling_seed": sampling_seed,
            "replace": replace,
            "num_available_starts": num_available_starts,
            "num_sampled_trajectories": num_sampled_trajectories,
            "unique_start_count": int(len(unique_starts)),
            "unique_start_ratio": float(len(unique_starts) / len(sampled_start_indices)),
            "duplicate_ratio": duplicate_ratio,
            "start_index_min": int(np.min(sampled_start_indices)),
            "start_index_max": int(np.max(sampled_start_indices)),
        }
    else:
        report["sampling"] = {
            "has_sampling_metadata": False,
        }
        report["soft_warnings"].append("dp_trajectories 未保存 sampled_start_indices，部分重放验证无法进行")

    # 约束统计
    single_trade = validate_single_trade_constraint(actions)
    report["single_trade_constraint"] = single_trade
    if single_trade["num_single_trade_violations"] > 0:
        report["hard_failures"].append(
            f"发现 {single_trade['num_single_trade_violations']} 条轨迹违反论文 single-trade 约束"
        )
    if single_trade["last_step_copy_violation_count"] > 0:
        report["hard_failures"].append(
            f"发现 {single_trade['last_step_copy_violation_count']} 条轨迹未满足 â[N-1] ← â[N-2]"
        )

    # 轨迹分布统计
    trajectory_returns = np.sum(rewards, axis=1).astype(np.float64)
    all_flat_mask = np.all(actions == DPPlanner.FLAT_ACTION, axis=1)
    token_action_hist = _histogram_from_int_array(actions.reshape(-1).astype(np.int64))
    report["trajectory_stats"] = {
        "all_flat_ratio": float(np.mean(all_flat_mask)) if all_flat_mask.size else 0.0,
        "non_flat_trajectory_ratio": float(np.mean(~all_flat_mask)) if all_flat_mask.size else 0.0,
        "token_action_histogram": token_action_hist,
        "trajectory_return_mean": _safe_mean(trajectory_returns),
        "trajectory_return_std": _safe_std(trajectory_returns),
        "trajectory_return_min": _safe_min(trajectory_returns),
        "trajectory_return_max": _safe_max(trajectory_returns),
    }
    if report["trajectory_stats"]["all_flat_ratio"] > 0.98:
        report["soft_warnings"].append("all_flat_ratio 过高，DP 轨迹可能退化为几乎全 flat")

    # 重放 / 重规划 / Bellman 子样本验证
    planner = DPPlanner(
        env=env,
        gamma=config.discount_factor,
        result_dir=config.result_dir,
        sampling_seed=config.phase1_sampling_seed,
    )
    subset_indices = _build_check_indices(int(states.shape[0]), dp_check_limit)
    replay_abs_errors: List[float] = []
    return_abs_errors: List[float] = []
    bellman_max_values: List[float] = []
    bellman_mean_values: List[float] = []
    state_match_count = 0
    action_match_count = 0
    checked_count = 0

    for subset_rank, traj_idx in enumerate(subset_indices):
        if sampled_start_indices is None:
            break
        start_index = int(sampled_start_indices[traj_idx])
        if start_index < 0:
            continue
        end_index = start_index + config.horizon
        price_end = min(end_index + 1, len(env.prices))
        h_states = env.states_dataframe.slice(start_index, config.horizon)
        h_prices = env.prices[start_index:price_end]

        saved_states = states[traj_idx]
        saved_actions = actions[traj_idx].astype(np.int64)
        saved_rewards = rewards[traj_idx].astype(np.float64)

        replanned_states, replanned_actions, replanned_rewards, debug_info = planner.plan(
            h_states,
            h_prices,
            return_debug_info=True,
        )
        replayed_rewards = replay_rewards_from_actions(env, start_index, saved_actions)

        state_match = bool(np.allclose(saved_states, replanned_states, atol=1e-6, rtol=1e-6))
        action_match = bool(np.array_equal(saved_actions, replanned_actions))
        reward_match = bool(np.allclose(saved_rewards, replanned_rewards, atol=1e-6, rtol=1e-6))
        replay_errors = np.abs(saved_rewards - replayed_rewards)
        bellman_stats = compute_bellman_residuals(planner, env, h_states, h_prices, debug_info)

        checked_count += 1
        state_match_count += int(state_match)
        action_match_count += int(action_match)
        replay_abs_errors.extend(replay_errors.tolist())
        return_abs_errors.append(float(abs(np.sum(saved_rewards) - np.sum(replayed_rewards))))
        bellman_max_values.append(float(bellman_stats["bellman_residual_max"]))
        bellman_mean_values.append(float(bellman_stats["bellman_residual_mean"]))

        if (not state_match or not action_match or not reward_match) and len(report["bad_case_examples"]) < 5:
            report["bad_case_examples"].append(
                {
                    "trajectory_index": int(traj_idx),
                    "subset_rank": int(subset_rank),
                    "start_index": start_index,
                    "state_match": state_match,
                    "action_match": action_match,
                    "reward_match": reward_match,
                    "reward_max_abs_error": float(np.max(np.abs(saved_rewards - replanned_rewards))),
                    "replay_reward_max_abs_error": float(np.max(replay_errors)),
                    "bellman_residual_max": bellman_stats["bellman_residual_max"],
                }
            )

    if checked_count == 0:
        report["soft_warnings"].append("未执行 DP 子样本 replay / replan 验证；请确认 sampled_start_indices 已保存")

    reward_replay_report = {
        "checked_trajectory_count": int(checked_count),
        "reward_mae": _safe_mean(np.asarray(replay_abs_errors, dtype=np.float64)),
        "reward_max_abs_error": _safe_max(np.asarray(replay_abs_errors, dtype=np.float64)),
        "return_abs_error_mean": _safe_mean(np.asarray(return_abs_errors, dtype=np.float64)),
        "return_abs_error_max": _safe_max(np.asarray(return_abs_errors, dtype=np.float64)),
        "state_replan_match_ratio": float(state_match_count / checked_count) if checked_count else 0.0,
        "policy_trace_match_ratio": float(action_match_count / checked_count) if checked_count else 0.0,
    }
    report["reward_replay"] = reward_replay_report
    report["bellman_consistency"] = {
        "checked_trajectory_count": int(checked_count),
        "bellman_residual_mean": _safe_mean(np.asarray(bellman_mean_values, dtype=np.float64)),
        "bellman_residual_max": _safe_max(np.asarray(bellman_max_values, dtype=np.float64)),
    }

    if checked_count > 0 and reward_replay_report["reward_max_abs_error"] > 1e-6:
        report["hard_failures"].append(
            f"reward 回放与保存值不一致，最大绝对误差={reward_replay_report['reward_max_abs_error']:.6e}"
        )
    if checked_count > 0 and reward_replay_report["policy_trace_match_ratio"] < 1.0:
        report["hard_failures"].append(
            f"DP 前向追踪与保存动作不一致，match_ratio={reward_replay_report['policy_trace_match_ratio']:.4f}"
        )
    if checked_count > 0 and report["bellman_consistency"]["bellman_residual_max"] > 1e-6:
        report["hard_failures"].append(
            f"Bellman residual 过大，max={report['bellman_consistency']['bellman_residual_max']:.6e}"
        )

    return report


def _compute_confusion_matrix(targets: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    """计算混淆矩阵。"""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    flat_targets = targets.reshape(-1)
    flat_preds = preds.reshape(-1)
    for true_label, pred_label in zip(flat_targets, flat_preds):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def _compute_classification_report(confusion: np.ndarray) -> Dict[str, Dict[str, float]]:
    """根据混淆矩阵计算每类 precision / recall / f1。"""
    report: Dict[str, Dict[str, float]] = {}
    for class_idx in range(confusion.shape[0]):
        tp = float(confusion[class_idx, class_idx])
        fp = float(np.sum(confusion[:, class_idx]) - tp)
        fn = float(np.sum(confusion[class_idx, :]) - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        report[str(class_idx)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return report


def _load_phase1_models(checkpoint: Dict[str, Any], device: torch.device) -> Tuple[VQEncoder, VQCodebook, VQDecoder, Dict[str, Any]]:
    """根据 checkpoint 中保存的配置还原 encoder / codebook / decoder。"""
    model_config = checkpoint.get("config", {})
    encoder = VQEncoder(
        state_dim=int(model_config["state_dim"]),
        action_dim=int(model_config["action_dim"]),
        hidden_dim=int(model_config["lstm_hidden_dim"]),
        latent_dim=int(model_config["latent_dim"]),
    ).to(device)
    codebook = VQCodebook(
        num_codes=int(model_config["num_archetypes"]),
        code_dim=int(model_config["latent_dim"]),
    ).to(device)
    decoder = VQDecoder(
        state_dim=int(model_config["state_dim"]),
        code_dim=int(model_config["latent_dim"]),
        hidden_dim=int(model_config["lstm_hidden_dim"]),
        action_dim=int(model_config["action_dim"]),
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    codebook.load_state_dict(checkpoint["codebook"], strict=True)
    decoder.load_state_dict(checkpoint["decoder"], strict=True)
    return encoder, codebook, decoder, model_config


def validate_phase1_model(
    config: Any,
    pair: str,
    trajectory_path: str,
    model_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    """验证 Phase I VQ 模型的 checkpoint、重建效果与 codebook 健康度。"""
    report: Dict[str, Any] = {
        "checkpoint_integrity": {},
        "forward_shape": {},
        "loss_decomposition": {},
        "reconstruction": {},
        "codebook_usage": {},
        "latent_geometry": {},
        "training_monitor_summary": {},
        "bad_case_examples": [],
        "hard_failures": [],
        "soft_warnings": [],
    }

    checkpoint = torch.load(model_path, map_location=device)
    available_keys = sorted(checkpoint.keys())
    required_keys = {"encoder", "codebook", "decoder", "config"}
    missing_keys = sorted(required_keys - set(checkpoint.keys()))
    report["checkpoint_integrity"] = {
        "available_keys": available_keys,
        "missing_required_keys": missing_keys,
        "has_loss_history": "loss_history" in checkpoint,
        "loss_history_length": len(checkpoint.get("loss_history", [])),
    }
    if missing_keys:
        report["hard_failures"].append(f"phase1_model checkpoint 缺少必要键: {missing_keys}")
        return report

    encoder, codebook, decoder, model_config = _load_phase1_models(checkpoint, device)
    encoder.eval()
    codebook.eval()
    decoder.eval()

    report["checkpoint_integrity"].update(
        {
            "state_dim_match": int(model_config["state_dim"]) == int(config.state_dim),
            "action_dim_match": int(model_config["action_dim"]) == int(config.action_dim),
            "latent_dim_match": int(model_config["latent_dim"]) == int(config.latent_dim),
            "num_archetypes_match": int(model_config["num_archetypes"]) == int(config.num_archetypes),
        }
    )
    if not report["checkpoint_integrity"]["state_dim_match"]:
        report["hard_failures"].append("checkpoint.state_dim 与当前 config 不一致")
    if not report["checkpoint_integrity"]["action_dim_match"]:
        report["hard_failures"].append("checkpoint.action_dim 与当前 config 不一致")

    dataset = TrajectoryDataset.from_npz(trajectory_path)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, drop_last=False)
    ce_loss_fn = nn.CrossEntropyLoss()

    total_samples = 0
    total_tokens = 0
    token_correct = 0
    exact_match_count = 0
    change_detect_correct = 0
    change_step_errors: List[float] = []
    loss_weight_sum = 0.0
    rec_loss_weight_sum = 0.0
    codebook_loss_weight_sum = 0.0
    encoder_commit_weight_sum = 0.0
    code_counts = np.zeros(int(model_config["num_archetypes"]), dtype=np.int64)
    confusion = np.zeros((int(model_config["action_dim"]), int(model_config["action_dim"])), dtype=np.int64)
    z_e_norm_values: List[np.ndarray] = []
    z_q_norm_values: List[np.ndarray] = []
    quantization_distance_values: List[np.ndarray] = []
    logit_abs_max = 0.0
    nearest_neighbor_match_ratio = 0.0
    forward_shapes_recorded = False

    with torch.no_grad():
        for batch_idx, (s_demo, a_demo, r_demo) in enumerate(dataloader):
            s_demo = s_demo.to(device)
            a_demo = a_demo.to(device)
            r_demo = r_demo.to(device)
            batch_size = int(s_demo.shape[0])
            batch_tokens = int(a_demo.numel())

            z_e = encoder(s_demo, a_demo, r_demo)
            z_q_st, indices, commitment_loss = codebook.quantize(z_e)
            action_logits = decoder(s_demo, z_q_st)
            preds = torch.argmax(action_logits, dim=-1)

            logits_flat = action_logits.reshape(-1, int(model_config["action_dim"]))
            targets_flat = a_demo.reshape(-1)
            rec_loss = ce_loss_fn(logits_flat, targets_flat)
            z_q_detached = z_q_st.detach()
            encoder_commitment = float(model_config.get("vq_beta0", config.vq_beta0)) * torch.mean((z_e - z_q_detached) ** 2)
            total_loss = rec_loss + commitment_loss + encoder_commitment

            total_samples += batch_size
            total_tokens += batch_tokens
            token_correct += int((preds == a_demo).sum().item())
            exact_match_count += int(torch.all(preds == a_demo, dim=1).sum().item())
            change_detect_correct += int(
                (((preds[:, 1:] != preds[:, :-1]).any(dim=1)) == ((a_demo[:, 1:] != a_demo[:, :-1]).any(dim=1))).sum().item()
            )
            loss_weight_sum += float(total_loss.item()) * batch_size
            rec_loss_weight_sum += float(rec_loss.item()) * batch_size
            codebook_loss_weight_sum += float(commitment_loss.item()) * batch_size
            encoder_commit_weight_sum += float(encoder_commitment.item()) * batch_size
            code_counts += np.bincount(indices.detach().cpu().numpy(), minlength=code_counts.shape[0])
            confusion += _compute_confusion_matrix(
                a_demo.detach().cpu().numpy(),
                preds.detach().cpu().numpy(),
                num_classes=int(model_config["action_dim"]),
            )
            z_e_norm_values.append(torch.norm(z_e, dim=1).detach().cpu().numpy())
            z_q_norm_values.append(torch.norm(z_q_st, dim=1).detach().cpu().numpy())
            quantization_distance_values.append(torch.norm(z_e - z_q_st, dim=1).detach().cpu().numpy())
            logit_abs_max = max(logit_abs_max, float(action_logits.abs().max().item()))

            if not forward_shapes_recorded:
                distances = (
                    torch.sum(z_e ** 2, dim=1, keepdim=True)
                    - 2 * z_e @ codebook.embeddings.weight.t()
                    + torch.sum(codebook.embeddings.weight ** 2, dim=1, keepdim=False)
                )
                manual_indices = torch.argmin(distances, dim=1)
                nearest_neighbor_match_ratio = float((manual_indices == indices).float().mean().item())
                report["forward_shape"] = {
                    "batch_state_shape": list(s_demo.shape),
                    "batch_action_shape": list(a_demo.shape),
                    "batch_reward_shape": list(r_demo.shape),
                    "z_e_shape": list(z_e.shape),
                    "z_q_shape": list(z_q_st.shape),
                    "indices_shape": list(indices.shape),
                    "action_logits_shape": list(action_logits.shape),
                    "nearest_neighbor_match_ratio": nearest_neighbor_match_ratio,
                }
                forward_shapes_recorded = True

            gt_changes = (a_demo[:, 1:] != a_demo[:, :-1]).detach().cpu().numpy()
            pred_changes = (preds[:, 1:] != preds[:, :-1]).detach().cpu().numpy()
            gt_has_change = gt_changes.any(axis=1)
            pred_has_change = pred_changes.any(axis=1)
            for row_idx in range(batch_size):
                if gt_has_change[row_idx] and pred_has_change[row_idx]:
                    gt_step = int(np.argmax(gt_changes[row_idx]) + 1)
                    pred_step = int(np.argmax(pred_changes[row_idx]) + 1)
                    change_step_errors.append(abs(gt_step - pred_step))

            mismatch_mask = ~torch.all(preds == a_demo, dim=1)
            if mismatch_mask.any() and len(report["bad_case_examples"]) < 5:
                mismatch_rows = torch.nonzero(mismatch_mask, as_tuple=False).view(-1).tolist()
                for row_idx in mismatch_rows:
                    if len(report["bad_case_examples"]) >= 5:
                        break
                    local_pred = preds[row_idx].detach().cpu().numpy()
                    local_true = a_demo[row_idx].detach().cpu().numpy()
                    mismatch_positions = np.where(local_pred != local_true)[0]
                    report["bad_case_examples"].append(
                        {
                            "trajectory_index": int(batch_idx * dataloader.batch_size + row_idx),
                            "first_mismatch_step": int(mismatch_positions[0]) if mismatch_positions.size else -1,
                            "pred_change_count": int(np.sum(local_pred[1:] != local_pred[:-1])) if local_pred.size >= 2 else 0,
                            "true_change_count": int(np.sum(local_true[1:] != local_true[:-1])) if local_true.size >= 2 else 0,
                        }
                    )

    if total_samples <= 0 or total_tokens <= 0:
        report["hard_failures"].append("验证数据集为空，无法评估 Phase I 模型")
        return report

    flat_baseline_accuracy = float(confusion[1, :].sum() / total_tokens)
    token_accuracy = float(token_correct / total_tokens)
    exact_match_rate = float(exact_match_count / total_samples)
    change_detect_accuracy = float(change_detect_correct / total_samples)

    z_e_norm_array = np.concatenate(z_e_norm_values) if z_e_norm_values else np.empty(0, dtype=np.float64)
    z_q_norm_array = np.concatenate(z_q_norm_values) if z_q_norm_values else np.empty(0, dtype=np.float64)
    quantization_distance_array = np.concatenate(quantization_distance_values) if quantization_distance_values else np.empty(0, dtype=np.float64)

    total_loss_mean = loss_weight_sum / total_samples
    rec_loss_mean = rec_loss_weight_sum / total_samples
    codebook_loss_mean = codebook_loss_weight_sum / total_samples
    encoder_commit_mean = encoder_commit_weight_sum / total_samples
    decomposition_residual = abs(total_loss_mean - (rec_loss_mean + codebook_loss_mean + encoder_commit_mean))

    report["loss_decomposition"] = {
        "eval_total_loss": float(total_loss_mean),
        "eval_rec_loss": float(rec_loss_mean),
        "eval_codebook_loss": float(codebook_loss_mean),
        "eval_encoder_commitment": float(encoder_commit_mean),
        "loss_decomposition_residual": float(decomposition_residual),
    }
    if not math.isfinite(total_loss_mean):
        report["hard_failures"].append("eval_total_loss 不是有限值")
    if decomposition_residual > 1e-6:
        report["hard_failures"].append(
            f"loss 分解不一致，residual={decomposition_residual:.6e}"
        )
    if report["forward_shape"].get("nearest_neighbor_match_ratio", 0.0) < 1.0:
        report["hard_failures"].append("codebook quantize 的 indices 与手动 argmin 结果不一致")

    report["reconstruction"] = {
        "token_accuracy": token_accuracy,
        "trajectory_exact_match_rate": exact_match_rate,
        "change_detect_accuracy": change_detect_accuracy,
        "change_step_mae": _safe_mean(np.asarray(change_step_errors, dtype=np.float64)),
        "flat_baseline_accuracy": flat_baseline_accuracy,
        "confusion_matrix": confusion.tolist(),
        "per_class_report": _compute_classification_report(confusion),
    }
    if token_accuracy <= flat_baseline_accuracy + 0.01:
        report["soft_warnings"].append(
            "token_accuracy 仅略高于 flat baseline，模型可能主要在预测 flat"
        )

    codebook_entropy = _compute_entropy_from_counts(code_counts)
    codebook_perplexity = float(math.exp(codebook_entropy)) if codebook_entropy > 0 else 1.0
    used_code_count = int(np.sum(code_counts > 0))
    dead_code_count = int(np.sum(code_counts == 0))
    dominant_code_ratio = float(np.max(code_counts) / np.sum(code_counts)) if np.sum(code_counts) > 0 else 0.0
    report["codebook_usage"] = {
        "code_usage_histogram": {str(i): int(v) for i, v in enumerate(code_counts.tolist())},
        "used_code_count": used_code_count,
        "dead_code_count": dead_code_count,
        "dominant_code_ratio": dominant_code_ratio,
        "codebook_entropy": codebook_entropy,
        "codebook_perplexity": codebook_perplexity,
    }
    if used_code_count == 0:
        report["hard_failures"].append("codebook 未被任何样本使用")
    if dominant_code_ratio > 0.95:
        report["soft_warnings"].append("dominant_code_ratio > 0.95，存在明显 code collapse 风险")
    if dead_code_count > code_counts.shape[0] // 2:
        report["soft_warnings"].append("超过一半的 code 未被使用，codebook 利用率偏低")

    codebook_weight = codebook.embeddings.weight.detach().cpu()
    normalized_weight = torch.nn.functional.normalize(codebook_weight, dim=1)
    cosine_matrix = normalized_weight @ normalized_weight.t()
    l2_matrix = torch.cdist(codebook_weight, codebook_weight, p=2)
    off_diag_mask = ~torch.eye(codebook_weight.shape[0], dtype=torch.bool)
    off_diag_cos = cosine_matrix[off_diag_mask].numpy()
    off_diag_l2 = l2_matrix[off_diag_mask].numpy()
    report["latent_geometry"] = {
        "z_e_norm_mean": _safe_mean(z_e_norm_array),
        "z_e_norm_std": _safe_std(z_e_norm_array),
        "z_q_norm_mean": _safe_mean(z_q_norm_array),
        "z_q_norm_std": _safe_std(z_q_norm_array),
        "quantization_mse": float(np.mean(np.square(quantization_distance_array))) if quantization_distance_array.size else 0.0,
        "quantization_distance_p95": float(np.percentile(quantization_distance_array, 95)) if quantization_distance_array.size else 0.0,
        "pairwise_codebook_cosine_mean": _safe_mean(off_diag_cos),
        "pairwise_codebook_cosine_min": _safe_min(off_diag_cos),
        "pairwise_codebook_cosine_max": _safe_max(off_diag_cos),
        "pairwise_codebook_l2_mean": _safe_mean(off_diag_l2),
        "pairwise_codebook_l2_min": _safe_min(off_diag_l2),
        "pairwise_codebook_l2_max": _safe_max(off_diag_l2),
        "logit_abs_max": logit_abs_max,
    }
    if report["latent_geometry"]["pairwise_codebook_cosine_min"] > 0.95:
        report["soft_warnings"].append("pairwise_codebook_cosine_min 过高，多个 codebook 向量可能过于相似")

    training_monitor = checkpoint.get("training_monitor", {})
    report["training_monitor_summary"] = training_monitor

    return report


def save_phase1_validation_report(report_path: str, report: Dict[str, Any]) -> None:
    """保存 Phase I 验证报告到 JSON 文件。"""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(_to_serializable(report), fp, ensure_ascii=False, indent=2)
    logger.info("Phase I 验证报告已保存到 %s", report_path)


def validate_phase1_artifacts(
    config: Any,
    pair: str,
    trajectory_path: str | None = None,
    model_path: str | None = None,
    report_path: str | None = None,
    env: TradingEnv | None = None,
    device: torch.device | None = None,
    dp_check_limit: int = 256,
) -> Dict[str, Any]:
    """统一验证 Phase I 产物，并保存 phase1_validation_report.json。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trajectory_path = trajectory_path or DPPlanner.build_trajectory_cache_path(config.result_dir, pair)
    model_path = model_path or os.path.join(config.result_dir, pair, "phase1_archetype_discovery", f"{pair}_vq_model.pt")
    report_path = report_path or os.path.join(config.result_dir, pair, "phase1_archetype_discovery", "phase1_validation_report.json")

    report: Dict[str, Any] = {
        "pair": pair,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "trajectory_path": trajectory_path,
            "model_path": model_path,
            "report_path": report_path,
        },
        "status": {
            "dp_passed": False,
            "model_passed": False,
            "overall_passed": False,
            "hard_failures": [],
            "soft_warnings": [],
        },
        "dp_validation": {},
        "model_validation": {},
    }

    try:
        if env is None:
            logger.info("Phase I 验证: 重新加载训练特征与 TradingEnv")
            pipeline = FeaturePipeline(config.data_dir, pair)
            train_df, _, _ = pipeline.get_state_vector()
            train_prices_df, _, _ = pipeline.get_prices()
            env = TradingEnv(
                states=train_df.to_numpy(),
                prices=train_prices_df["close"].to_numpy(),
                pair=pair,
                horizon=config.horizon,
                states_dataframe=train_df,
                max_positions=config.max_positions,
                commission_rate=config.commission_rate,
            )

        dp_report = validate_dp_trajectories(
            config=config,
            pair=pair,
            env=env,
            trajectory_path=trajectory_path,
            dp_check_limit=dp_check_limit,
        )
        model_report = validate_phase1_model(
            config=config,
            pair=pair,
            trajectory_path=trajectory_path,
            model_path=model_path,
            device=device,
        )

        dp_hard_failures = list(dp_report.get("hard_failures", []))
        model_hard_failures = list(model_report.get("hard_failures", []))
        dp_soft_warnings = list(dp_report.get("soft_warnings", []))
        model_soft_warnings = list(model_report.get("soft_warnings", []))

        report["dp_validation"] = dp_report
        report["model_validation"] = model_report
        report["status"] = {
            "dp_passed": len(dp_hard_failures) == 0,
            "model_passed": len(model_hard_failures) == 0,
            "overall_passed": len(dp_hard_failures) == 0 and len(model_hard_failures) == 0,
            "hard_failures": dp_hard_failures + model_hard_failures,
            "soft_warnings": dp_soft_warnings + model_soft_warnings,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Phase I 验证过程中发生异常")
        report["status"] = {
            "dp_passed": False,
            "model_passed": False,
            "overall_passed": False,
            "hard_failures": [f"验证过程异常: {exc}"],
            "soft_warnings": [],
        }

    save_phase1_validation_report(report_path, report)
    return report
