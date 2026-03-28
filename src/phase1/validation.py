"""Phase I 验证工具 — DP 轨迹正确性 + VQ 原型聚合质量

# 论文关联:
# - Section 3.1 / Eq. 1: 逐步奖励 r_step_t = P_t × (p_{t+1} - p_t) - O_t
# - Algorithm 1: Single-trade DP planner
# - Section 4.1 / Eq. 1-4: Encoder / Codebook / Decoder / VQ 损失
#
# 本版本相对上一版的关键修正:
# 1. 回放一致性阈值改为浮点容差判断，而不是 1e-8 级别硬卡死。
# 2. VQ 重建指标改为使用 decoder.generate() 的真实 rollout，而不是 teacher forcing argmax。
# 3. 新增 dominant_code_ratio，直接暴露“主 code 吞噬样本”的程度。
# 4. 保留 DP oracle / replay / archetype 聚合三层验证结构。
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.env.trading_env import TradingEnv
from src.phase1.codebook import VQCodebook
from src.phase1.dp_planner import DPPlanner
from src.phase1.vq_decoder import VQDecoder
from src.phase1.vq_encoder import VQEncoder


@dataclass
class ValidationConfig:
    """Phase I 验证配置。"""

    batch_size: int = 256
    max_eval_samples: int = 512
    dp_oracle_num_horizons: int = 8
    dp_bruteforce_horizon: int = 6
    replay_num_trajectories: int = 512
    replay_abs_tol: float = 1e-5


# ---------------------------------------------------------------------------
# 通用基础工具
# ---------------------------------------------------------------------------


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0



def _entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts.astype(np.float64) / float(total)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())



def _perplexity_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts.astype(np.float64) / float(total)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    entropy = float(-(probs * np.log(probs)).sum())
    return float(np.exp(entropy))



def _sequence_hash(states: np.ndarray, actions: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(states).tobytes())
    h.update(np.ascontiguousarray(actions).tobytes())
    return h.hexdigest()



def _prepare_dataloader(states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(states).float(),
        torch.from_numpy(actions).long(),
        torch.from_numpy(rewards).float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


# ---------------------------------------------------------------------------
# DP 约束与 oracle 验证
# ---------------------------------------------------------------------------


def is_valid_single_trade_path(planner: DPPlanner, actions: Sequence[int]) -> bool:
    current_c = 0
    prev_action = planner.FLAT_ACTION
    for action in actions:
        current_c = planner._compute_next_constraint(current_c, prev_action, int(action))
        if current_c < 0:
            return False
        prev_action = int(action)
    return True



def count_constraint_violations(planner: DPPlanner, action_batch: np.ndarray) -> int:
    return int(sum(0 if is_valid_single_trade_path(planner, actions) else 1 for actions in action_batch))



def compute_discounted_return(
    planner: DPPlanner,
    states: pl.DataFrame,
    prices: np.ndarray,
    actions: Sequence[int],
) -> Tuple[np.ndarray, float]:
    horizon = len(actions)
    rewards = np.zeros(horizon, dtype=np.float64)

    current_position = 0
    discounted_return = 0.0

    for t, action in enumerate(actions):
        action = int(action)
        next_position = TradingEnv.POSITION_MAP[action] * planner.m
        p_t = float(prices[t])
        p_next = float(prices[t + 1]) if (t + 1) < len(prices) else float(prices[t])
        execution_cost = planner.env.compute_execution_cost(
            action,
            current_position,
            p_t,
            states.row(t, named=True),
        )
        reward = float(next_position * (p_next - p_t) - execution_cost)
        rewards[t] = reward
        discounted_return += (planner.gamma ** t) * reward
        current_position = next_position

    return rewards, float(discounted_return)



def brute_force_optimal_sequence(
    planner: DPPlanner,
    states: pl.DataFrame,
    prices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    horizon = len(states)
    best_actions = None
    best_rewards = None
    best_return = -np.inf

    for candidate in itertools.product(DPPlanner.ACTIONS, repeat=horizon):
        if not is_valid_single_trade_path(planner, candidate):
            continue
        rewards, discounted_return = compute_discounted_return(planner, states, prices, candidate)
        if discounted_return > best_return:
            best_return = discounted_return
            best_actions = np.asarray(candidate, dtype=np.int32)
            best_rewards = rewards

    if best_actions is None or best_rewards is None:
        raise RuntimeError("未找到满足约束的可行动作序列")

    return best_actions, best_rewards, float(best_return)



def evaluate_dp_bruteforce_oracle(
    planner: DPPlanner,
    states_dataframe: pl.DataFrame,
    prices: np.ndarray,
    horizon: int,
    oracle_num_horizons: int,
    bruteforce_horizon: int,
) -> Dict[str, Any]:
    num_horizons = min(len(states_dataframe) // horizon, oracle_num_horizons)
    details: List[Dict[str, Any]] = []
    gaps: List[float] = []
    action_match_flags: List[bool] = []

    for h_idx in range(num_horizons):
        start = h_idx * horizon
        end = start + bruteforce_horizon
        sub_states = states_dataframe[start:end]
        sub_prices = prices[start : min(end + 1, len(prices))]

        if len(sub_states) < bruteforce_horizon:
            break

        _, dp_actions, dp_rewards = planner.plan(sub_states, sub_prices)
        dp_discounted_return = sum((planner.gamma ** t) * float(r) for t, r in enumerate(dp_rewards))
        brute_actions, brute_rewards, brute_return = brute_force_optimal_sequence(planner, sub_states, sub_prices)

        gap = float(brute_return - dp_discounted_return)
        gaps.append(gap)
        action_match = bool(np.array_equal(dp_actions, brute_actions))
        action_match_flags.append(action_match)

        details.append(
            {
                "horizon_idx": int(h_idx),
                "dp_actions": dp_actions.tolist(),
                "bruteforce_actions": brute_actions.tolist(),
                "dp_discounted_return": float(dp_discounted_return),
                "bruteforce_discounted_return": float(brute_return),
                "optimality_gap": gap,
                "action_match": action_match,
                "dp_rewards": np.asarray(dp_rewards).tolist(),
                "bruteforce_rewards": np.asarray(brute_rewards).tolist(),
            }
        )

    return {
        "num_cases": len(details),
        "bruteforce_horizon": int(bruteforce_horizon),
        "max_optimality_gap": float(max(gaps)) if gaps else 0.0,
        "mean_optimality_gap": _safe_mean(gaps),
        "all_optimal": bool(all(abs(g) < 1e-8 for g in gaps)) if gaps else True,
        "action_match_rate": _safe_mean([1.0 if x else 0.0 for x in action_match_flags]),
        "details": details,
    }


# ---------------------------------------------------------------------------
# 轨迹回放一致性验证
# ---------------------------------------------------------------------------


def replay_actions_on_horizon(env: TradingEnv, horizon_idx: int, actions: Sequence[int]) -> np.ndarray:
    rewards: List[float] = []
    env.reset(horizon_idx)
    for action in actions:
        _, reward, _, _ = env.step(int(action))
        rewards.append(float(reward))
    return np.asarray(rewards, dtype=np.float64)



def validate_trajectory_replay(
    planner: DPPlanner,
    env: TradingEnv,
    trajectories: Dict[str, np.ndarray],
    num_trajectories: int,
    abs_tol: float,
) -> Dict[str, Any]:
    actions = trajectories["actions"]
    rewards = trajectories["rewards"]
    horizon_indices = trajectories.get("horizon_indices")

    limit = min(len(actions), num_trajectories)
    num_horizons = max(env.num_horizons, 1)

    maes: List[float] = []
    max_abs_list: List[float] = []
    details: List[Dict[str, Any]] = []

    for sample_idx in range(limit):
        # 兼容两种轨迹来源：
        # 1. 旧版按 horizon 顺序循环复用 -> sample_idx % num_horizons
        # 2. 论文对齐版随机采样 chunk -> 显式保存 horizon_indices
        horizon_idx = int(horizon_indices[sample_idx]) if horizon_indices is not None else (sample_idx % num_horizons)
        replay_rewards = replay_actions_on_horizon(env, horizon_idx, actions[sample_idx])
        target_rewards = rewards[sample_idx].astype(np.float64)
        abs_diff = np.abs(replay_rewards - target_rewards)
        maes.append(float(abs_diff.mean()))
        max_abs_list.append(float(abs_diff.max()))

        if sample_idx < 5:
            details.append(
                {
                    "sample_idx": int(sample_idx),
                    "horizon_idx": int(horizon_idx),
                    "actions": actions[sample_idx].tolist(),
                    "saved_rewards": target_rewards.tolist(),
                    "replay_rewards": replay_rewards.tolist(),
                    "mae": float(abs_diff.mean()),
                    "max_abs": float(abs_diff.max()),
                }
            )

    replay_max_abs = float(max(max_abs_list)) if max_abs_list else 0.0
    return {
        "num_trajectories": int(limit),
        "reward_replay_mae": _safe_mean(maes),
        "reward_replay_max_abs": replay_max_abs,
        "allclose": bool(replay_max_abs <= abs_tol),
        "constraint_violation_count": int(count_constraint_violations(planner, actions[:limit])),
        "abs_tol": float(abs_tol),
        "details": details,
    }


# ---------------------------------------------------------------------------
# VQ 重建与 archetype 聚合质量验证
# ---------------------------------------------------------------------------


def _compute_template_purity(sequences: np.ndarray) -> float:
    if len(sequences) == 0:
        return 0.0
    counter = Counter(tuple(map(int, seq)) for seq in sequences)
    return float(counter.most_common(1)[0][1] / len(sequences))



def _compute_within_code_entropy(sequences: np.ndarray, action_dim: int = 3) -> float:
    if len(sequences) == 0:
        return 0.0
    horizon = sequences.shape[1]
    entropies = []
    for t in range(horizon):
        counts = np.bincount(sequences[:, t].astype(np.int64), minlength=action_dim)
        entropies.append(_entropy_from_counts(counts))
    return float(np.mean(entropies)) if entropies else 0.0



def _duplicate_sample_ratio(states: np.ndarray, actions: np.ndarray) -> float:
    if len(states) == 0:
        return 0.0
    hashes = [_sequence_hash(states[i], actions[i]) for i in range(len(states))]
    unique_count = len(set(hashes))
    return float(1.0 - unique_count / len(hashes))



def _build_reconstruction_metrics(
    planner: DPPlanner,
    env: TradingEnv,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    pred_actions: np.ndarray,
    code_indices: np.ndarray,
    codebook_size: int,
    max_eval_samples: int,
    horizon_indices: np.ndarray | None = None,
) -> Dict[str, Any]:
    step_acc = float((pred_actions == actions).mean())
    exact_seq_match = float(np.mean(np.all(pred_actions == actions, axis=1)))

    num_horizons = max(env.num_horizons, 1)
    reward_retention: List[float] = []
    reward_regret: List[float] = []
    eval_count = min(len(pred_actions), max_eval_samples)

    for sample_idx in range(eval_count):
        horizon_idx = int(horizon_indices[sample_idx]) if horizon_indices is not None else (sample_idx % num_horizons)
        replay_pred = replay_actions_on_horizon(env, horizon_idx, pred_actions[sample_idx])
        demo_return = float(np.sum(rewards[sample_idx]))
        pred_return = float(np.sum(replay_pred))
        reward_retention.append(float(pred_return / (abs(demo_return) + 1e-8)))
        reward_regret.append(float(demo_return - pred_return))

    counts = np.bincount(code_indices.astype(np.int64), minlength=codebook_size)
    demo_returns = rewards.sum(axis=1)
    per_code: Dict[str, Any] = {}
    within_code_entropies: List[float] = []
    within_code_return_vars: List[float] = []
    template_purities: List[float] = []

    for code_id in range(codebook_size):
        mask = code_indices == code_id
        code_actions = pred_actions[mask]
        code_returns = demo_returns[mask]
        usage = int(mask.sum())
        entropy = _compute_within_code_entropy(code_actions) if usage > 0 else 0.0
        return_var = float(np.var(code_returns)) if usage > 0 else 0.0
        template_purity = _compute_template_purity(code_actions) if usage > 0 else 0.0
        within_code_entropies.append(entropy)
        within_code_return_vars.append(return_var)
        template_purities.append(template_purity)
        per_code[str(code_id)] = {
            "usage": usage,
            "usage_ratio": float(usage / len(actions)) if len(actions) > 0 else 0.0,
            "within_code_entropy": entropy,
            "within_code_return_var": return_var,
            "template_purity": template_purity,
            "mean_demo_return": float(np.mean(code_returns)) if usage > 0 else 0.0,
        }

    dominant_code_ratio = float(counts.max() / max(len(actions), 1))
    return {
        "num_samples": int(len(actions)),
        "step_accuracy": step_acc,
        "exact_sequence_match": exact_seq_match,
        "recon_constraint_violation_count": int(count_constraint_violations(planner, pred_actions)),
        "reward_retention_mean": _safe_mean(reward_retention),
        "reward_retention_median": float(np.median(reward_retention)) if reward_retention else 0.0,
        "reward_regret_mean": _safe_mean(reward_regret),
        "reward_regret_p95": float(np.percentile(reward_regret, 95)) if reward_regret else 0.0,
        "code_usage_histogram": counts.astype(int).tolist(),
        "code_perplexity": _perplexity_from_counts(counts),
        "dominant_code_ratio": dominant_code_ratio,
        "used_code_count": int((counts > 0).sum()),
        "dead_code_count": int((counts == 0).sum()),
        "within_code_entropy_mean": _safe_mean(within_code_entropies),
        "within_code_return_var_mean": _safe_mean(within_code_return_vars),
        "template_purity_mean": _safe_mean(template_purities),
        "duplicate_sample_ratio": _duplicate_sample_ratio(states, actions),
        "per_code": per_code,
    }



def evaluate_vq_reconstruction(
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
    planner: DPPlanner,
    env: TradingEnv,
    trajectories: Dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    max_eval_samples: int,
) -> Dict[str, Any]:
    states = trajectories["states"].astype(np.float32)
    actions = trajectories["actions"].astype(np.int64)
    rewards = trajectories["rewards"].astype(np.float32)
    horizon_indices = trajectories.get("horizon_indices")
    if horizon_indices is not None:
        horizon_indices = horizon_indices.astype(np.int64)

    dataloader = _prepare_dataloader(states, actions, rewards, batch_size=batch_size)
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    encoder.eval()
    codebook.eval()
    decoder.eval()

    pred_all: List[np.ndarray] = []
    idx_all: List[np.ndarray] = []
    ce_losses: List[float] = []
    teacher_forcing_step_acc: List[float] = []

    with torch.no_grad():
        for s_demo, a_demo, r_demo in dataloader:
            s_demo = s_demo.to(device)
            a_demo = a_demo.to(device)
            r_demo = r_demo.to(device)

            z_e = encoder(s_demo, a_demo, r_demo)
            z_q_st, indices, _ = codebook.quantize(z_e)

            # teacher forcing 指标: 看模型在已知上一动作条件下的局部分类质量。
            teacher_logits = decoder(s_demo, z_q_st, teacher_actions=a_demo)
            logits_flat = teacher_logits.reshape(-1, decoder.action_dim)
            targets_flat = a_demo.reshape(-1)
            ce_losses.append(float(ce_loss_fn(logits_flat, targets_flat).item()))
            teacher_preds = torch.argmax(teacher_logits, dim=-1)
            teacher_forcing_step_acc.append(float((teacher_preds == a_demo).float().mean().item()))

            # rollout 指标: 用 decoder.generate() 的自回归输出评估真正的可执行重建质量。
            rollout_actions, _ = decoder.generate(s_demo, z_q_st)
            pred_all.append(rollout_actions.detach().cpu().numpy())
            idx_all.append(indices.detach().cpu().numpy())

    pred_actions_np = np.concatenate(pred_all, axis=0) if pred_all else np.empty_like(actions)
    code_indices_np = np.concatenate(idx_all, axis=0) if idx_all else np.zeros(len(actions), dtype=np.int64)

    metrics = _build_reconstruction_metrics(
        planner=planner,
        env=env,
        states=states,
        actions=actions,
        rewards=rewards,
        pred_actions=pred_actions_np,
        code_indices=code_indices_np,
        codebook_size=codebook.num_codes,
        max_eval_samples=max_eval_samples,
        horizon_indices=horizon_indices,
    )
    metrics["reconstruction_ce_loss"] = _safe_mean(ce_losses)
    metrics["teacher_forcing_step_accuracy"] = _safe_mean(teacher_forcing_step_acc)
    return metrics


# ---------------------------------------------------------------------------
# 高层总入口
# ---------------------------------------------------------------------------


def run_phase1_validation(
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
    planner: DPPlanner,
    env: TradingEnv,
    states_dataframe: pl.DataFrame,
    prices: np.ndarray,
    trajectories: Dict[str, np.ndarray],
    config: ValidationConfig,
    device: torch.device | str,
) -> Dict[str, Any]:
    device = torch.device(device)

    replay_report = validate_trajectory_replay(
        planner=planner,
        env=env,
        trajectories=trajectories,
        num_trajectories=config.replay_num_trajectories,
        abs_tol=config.replay_abs_tol,
    )

    dp_oracle_report = evaluate_dp_bruteforce_oracle(
        planner=planner,
        states_dataframe=states_dataframe,
        prices=prices,
        horizon=env.horizon,
        oracle_num_horizons=config.dp_oracle_num_horizons,
        bruteforce_horizon=config.dp_bruteforce_horizon,
    )

    vq_report = evaluate_vq_reconstruction(
        encoder=encoder,
        codebook=codebook,
        decoder=decoder,
        planner=planner,
        env=env,
        trajectories=trajectories,
        batch_size=config.batch_size,
        device=device,
        max_eval_samples=config.max_eval_samples,
    )

    hard_pass = (
        replay_report["allclose"]
        and replay_report["constraint_violation_count"] == 0
        and dp_oracle_report["all_optimal"]
    )

    phase2_ready = bool(
        hard_pass
        and vq_report["step_accuracy"] >= 0.60
        and vq_report["reward_retention_mean"] > 0.0
        and vq_report["code_perplexity"] >= 2.0
        and vq_report["dominant_code_ratio"] <= 0.60
    )

    return {
        "summary": {
            "hard_validation_passed": bool(hard_pass),
            "phase1_ready_for_phase2": phase2_ready,
        },
        "replay_validation": replay_report,
        "dp_oracle_validation": dp_oracle_report,
        "vq_validation": vq_report,
    }



def save_validation_report(report: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
