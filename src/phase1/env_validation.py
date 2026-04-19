"""Phase I 环境级验证模块 — 验证 archetype 在真实交易环境中的表现

现有 validation.py 侧重于 VQ 重建精度（token accuracy / exact match）和
codebook 几何健康度，但这些指标无法直接反映 archetype 在 Phase II 中的实际
可用性。本模块补充以下关键诊断：

1. 每个 archetype 的环境级 return 分布 — 哪些 archetype 有正收益
2. Decoder 重建动作 vs DP 原始动作在环境中的 return 差距
3. Archetype 之间的行为差异性 — 10 个 archetype 是否产生不同策略
4. 验证集上的泛化表现 — 训练集 token accuracy 高但验证集可能退化
5. Decoder 动作分布偏移 — 重建偏差在环境中的放大效应
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import TrajectoryDataset
from src.env.trading_env import TradingEnv
from src.phase1.codebook import VQCodebook
from src.phase1.vq_decoder import VQDecoder
from src.phase1.vq_encoder import VQEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size > 0 else 0.0


def _safe_std(arr: np.ndarray) -> float:
    return float(np.std(arr)) if arr.size > 0 else 0.0


def _safe_percentile(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if arr.size > 0 else 0.0


def _run_actions_in_env(
    env: TradingEnv,
    horizon_idx: int,
    actions: np.ndarray,
) -> Dict[str, Any]:
    """在环境中执行一段动作序列，返回 return 和执行细节。"""
    env.reset(horizon_idx)
    total_return = 0.0
    total_cost = 0.0
    gross_pnl = 0.0
    position_changes = 0

    for step_idx in range(len(actions)):
        action = int(actions[step_idx])
        _, reward, done, info = env.step(action)
        cost = float(info.get("execution_cost", 0.0))
        old_pos = int(info.get("old_position", 0))
        new_pos = int(info.get("position", old_pos))
        total_return += reward
        total_cost += cost
        gross_pnl += reward + cost
        if old_pos != new_pos:
            position_changes += 1
        if done:
            break

    return {
        "total_return": total_return,
        "gross_pnl": gross_pnl,
        "total_cost": total_cost,
        "position_changes": position_changes,
    }


def _decode_horizon(
    decoder: VQDecoder,
    z_q: torch.Tensor,
    horizon_states: np.ndarray,
    device: torch.device,
    normalizer: TrajectoryDataset | None = None,
) -> np.ndarray:
    """用 frozen decoder 生成一个 horizon 的动作序列。"""
    if horizon_states.ndim != 2 or horizon_states.shape[1] != decoder.state_dim:
        raise ValueError(
            "decoder 输入状态维度不匹配: "
            f"actual={tuple(horizon_states.shape)}, expected=(*, {decoder.state_dim})"
        )
    if normalizer is not None:
        horizon_states = normalizer.normalize_states(horizon_states)
    states_t = torch.tensor(
        horizon_states, dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        actions = decoder.decode_with_single_trade_constraint(states_t, z_q).squeeze(0)
    return actions.cpu().numpy()


# ---------------------------------------------------------------------------
# 1. 每个 Archetype 的环境级 Return 分布
# ---------------------------------------------------------------------------

def validate_archetype_env_returns(
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
    env: TradingEnv,
    trajectory_dataset: TrajectoryDataset,
    device: torch.device,
    max_horizons: int = 512,
) -> Dict[str, Any]:
    """评估每个 archetype 在真实环境中的 return 分布。

    对训练集中的 horizon 子集：
    - 用 encoder+codebook 得到 ground-truth archetype label
    - 用 frozen decoder + 对应 archetype embedding 生成动作
    - 在 env 中执行，统计每个 archetype 的 return 分布

    这是诊断 Phase 2 失败的最关键指标：如果所有 archetype 的 env return
    都是负的，Phase 2 的 selector 无论怎么选都不会好。
    """
    encoder.eval()
    codebook.eval()
    decoder.eval()

    K = codebook.num_codes
    num_horizons = min(env.num_horizons, max_horizons, len(trajectory_dataset))
    h = env.horizon

    # 收集每个 archetype 的 env return
    archetype_returns: Dict[int, List[float]] = {k: [] for k in range(K)}
    archetype_costs: Dict[int, List[float]] = {k: [] for k in range(K)}
    dp_returns: List[float] = []
    decoded_returns: List[float] = []
    return_gaps: List[float] = []  # dp_return - decoded_return

    indices_to_check = np.random.choice(num_horizons, size=min(num_horizons, max_horizons), replace=False)

    # 判断 dataset 是否携带滑窗真实起点（由 DPPlanner 写入 npz 的 sampled_start_indices）
    has_start_indices = (
        hasattr(trajectory_dataset, "sampled_start_indices")
        and trajectory_dataset.sampled_start_indices is not None
    )

    for idx in indices_to_check:
        idx = int(idx)
        s_demo, a_demo, r_demo = trajectory_dataset[idx]
        s_demo_t = torch.as_tensor(s_demo, dtype=torch.float32, device=device).unsqueeze(0)
        a_demo_t = torch.as_tensor(a_demo, dtype=torch.long, device=device).unsqueeze(0)
        r_demo_t = torch.as_tensor(r_demo, dtype=torch.float32, device=device).unsqueeze(0)

        # 获取 ground-truth archetype label
        with torch.no_grad():
            z_e = encoder(s_demo_t, a_demo_t, r_demo_t)
            _, gt_idx, _ = codebook.quantize(z_e)
            k = int(gt_idx.item())

        # 确定该样本在原始时间序列中的真实起点
        # 若 dataset 携带 sampled_start_indices（滑窗采样），用真实 start 换算
        # horizon_idx，避免 idx * h 与滑窗坐标系不一致的 bug。
        if has_start_indices:
            true_start = int(trajectory_dataset.sampled_start_indices[idx])
            horizon_idx = true_start // h  # 映射到最近的非重叠 horizon
        else:
            true_start = idx * h
            horizon_idx = idx

        # 用 DP 原始动作在 env 中执行
        if horizon_idx < env.num_horizons:
            dp_result = _run_actions_in_env(env, horizon_idx, a_demo.numpy() if hasattr(a_demo, 'numpy') else a_demo)
            dp_returns.append(dp_result["total_return"])

            # 用 decoder 重建动作在 env 中执行（状态从真实 start 取）
            end = min(true_start + h, len(env.states))
            horizon_states = env.states[true_start:end]
            z_q = codebook.embeddings.weight[k].unsqueeze(0)
            decoded_actions = _decode_horizon(
                decoder, z_q, horizon_states, device,
                normalizer=trajectory_dataset,
            )
            dec_result = _run_actions_in_env(env, horizon_idx, decoded_actions)

            archetype_returns[k].append(dec_result["total_return"])
            archetype_costs[k].append(dec_result["total_cost"])
            decoded_returns.append(dec_result["total_return"])
            return_gaps.append(dp_result["total_return"] - dec_result["total_return"])

    # 汇总
    dp_arr = np.array(dp_returns, dtype=np.float64)
    dec_arr = np.array(decoded_returns, dtype=np.float64)
    gap_arr = np.array(return_gaps, dtype=np.float64)

    per_archetype = {}
    positive_archetype_count = 0
    for k in range(K):
        returns_k = np.array(archetype_returns[k], dtype=np.float64)
        costs_k = np.array(archetype_costs[k], dtype=np.float64)
        avg_ret = _safe_mean(returns_k)
        if avg_ret > 0:
            positive_archetype_count += 1
        per_archetype[str(k)] = {
            "count": len(archetype_returns[k]),
            "return_mean": avg_ret,
            "return_std": _safe_std(returns_k),
            "return_median": _safe_percentile(returns_k, 50),
            "return_p25": _safe_percentile(returns_k, 25),
            "return_p75": _safe_percentile(returns_k, 75),
            "positive_ratio": float(np.mean(returns_k > 0)) if returns_k.size > 0 else 0.0,
            "avg_cost": _safe_mean(costs_k),
        }

    report = {
        "num_horizons_checked": len(indices_to_check),
        "dp_return_mean": _safe_mean(dp_arr),
        "dp_return_std": _safe_std(dp_arr),
        "decoded_return_mean": _safe_mean(dec_arr),
        "decoded_return_std": _safe_std(dec_arr),
        "return_gap_mean": _safe_mean(gap_arr),
        "return_gap_std": _safe_std(gap_arr),
        "return_gap_p95": _safe_percentile(np.abs(gap_arr), 95),
        "positive_archetype_count": positive_archetype_count,
        "per_archetype": per_archetype,
        "warnings": [],
    }

    if positive_archetype_count == 0:
        report["warnings"].append(
            "所有 archetype 的平均 env return 均为负，Phase 2 selector 无法从中选出正收益策略"
        )
    if _safe_mean(gap_arr) > abs(_safe_mean(dp_arr)) * 0.5 and dp_arr.size > 0:
        report["warnings"].append(
            f"decoder 重建导致的 return 损失过大: gap_mean={_safe_mean(gap_arr):.4f}, "
            f"dp_return_mean={_safe_mean(dp_arr):.4f}"
        )

    return report


# ---------------------------------------------------------------------------
# 2. Archetype 行为差异性分析
# ---------------------------------------------------------------------------

def validate_archetype_diversity(
    codebook: VQCodebook,
    decoder: VQDecoder,
    env: TradingEnv,
    device: torch.device,
    sample_horizons: int = 128,
    normalizer: TrajectoryDataset | None = None,
) -> Dict[str, Any]:
    """检查不同 archetype 是否真的产生不同的交易策略。

    如果多个 archetype 在相同市场状态下生成几乎相同的动作序列，
    说明 codebook 虽然有多个 code，但 decoder 无法区分它们，
    Phase 2 的 selector 选哪个都一样。
    """
    codebook.eval()
    decoder.eval()

    K = codebook.num_codes
    h = env.horizon
    num_horizons = min(env.num_horizons, sample_horizons)
    horizon_indices = np.random.choice(env.num_horizons, size=num_horizons, replace=False)

    # 对每个 horizon，用所有 K 个 archetype 生成动作，比较差异
    action_agreement_matrix = np.zeros((K, K), dtype=np.float64)
    action_agreement_counts = np.zeros((K, K), dtype=np.int64)
    per_archetype_action_dist = np.zeros((K, 3), dtype=np.float64)  # action {0,1,2} 分布
    per_archetype_total_tokens = np.zeros(K, dtype=np.int64)

    for h_idx in horizon_indices:
        start = int(h_idx) * h
        end = min(start + h, len(env.states))
        horizon_states = env.states[start:end]
        actual_h = len(horizon_states)

        all_actions = np.zeros((K, actual_h), dtype=np.int64)
        for k in range(K):
            z_q = codebook.embeddings.weight[k].unsqueeze(0)
            all_actions[k] = _decode_horizon(
                decoder, z_q, horizon_states, device,
                normalizer=normalizer,
            )
            for a in range(3):
                per_archetype_action_dist[k, a] += np.sum(all_actions[k] == a)
            per_archetype_total_tokens[k] += actual_h

        # 计算 pairwise action agreement
        for i in range(K):
            for j in range(i, K):
                agreement = float(np.mean(all_actions[i] == all_actions[j]))
                action_agreement_matrix[i, j] += agreement
                action_agreement_matrix[j, i] += agreement
                action_agreement_counts[i, j] += 1
                action_agreement_counts[j, i] += 1

    # 归一化
    safe_counts = np.maximum(action_agreement_counts, 1)
    avg_agreement = action_agreement_matrix / safe_counts

    # 提取 off-diagonal 统计
    off_diag_mask = ~np.eye(K, dtype=bool)
    off_diag_agreements = avg_agreement[off_diag_mask]

    # 归一化 action 分布
    safe_tokens = np.maximum(per_archetype_total_tokens, 1).reshape(-1, 1)
    action_dist_normalized = per_archetype_action_dist / safe_tokens

    # 计算 archetype 之间 action 分布的 JS divergence
    js_divergences = []
    for i in range(K):
        for j in range(i + 1, K):
            p = action_dist_normalized[i] + 1e-10
            q = action_dist_normalized[j] + 1e-10
            p = p / p.sum()
            q = q / q.sum()
            m = 0.5 * (p + q)
            kl_pm = float(np.sum(p * np.log(p / m)))
            kl_qm = float(np.sum(q * np.log(q / m)))
            js_divergences.append(0.5 * (kl_pm + kl_qm))

    report = {
        "num_horizons_sampled": num_horizons,
        "pairwise_action_agreement_mean": _safe_mean(off_diag_agreements),
        "pairwise_action_agreement_min": float(np.min(off_diag_agreements)) if off_diag_agreements.size > 0 else 0.0,
        "pairwise_action_agreement_max": float(np.max(off_diag_agreements)) if off_diag_agreements.size > 0 else 0.0,
        "pairwise_action_agreement_matrix": avg_agreement.tolist(),
        "per_archetype_action_distribution": {
            str(k): {
                "short": float(action_dist_normalized[k, 0]),
                "flat": float(action_dist_normalized[k, 1]),
                "long": float(action_dist_normalized[k, 2]),
            }
            for k in range(K)
        },
        "pairwise_js_divergence_mean": _safe_mean(np.array(js_divergences)),
        "pairwise_js_divergence_min": float(np.min(js_divergences)) if js_divergences else 0.0,
        "pairwise_js_divergence_max": float(np.max(js_divergences)) if js_divergences else 0.0,
        "warnings": [],
    }

    if _safe_mean(off_diag_agreements) > 0.95:
        report["warnings"].append(
            f"archetype 之间的动作一致性过高 (mean={_safe_mean(off_diag_agreements):.4f})，"
            "decoder 可能无法区分不同 archetype，Phase 2 选择无意义"
        )
    if _safe_mean(np.array(js_divergences)) < 0.01:
        report["warnings"].append(
            "archetype 之间的动作分布 JS divergence 极低，策略几乎无差异"
        )

    return report


# ---------------------------------------------------------------------------
# 3. 验证集泛化表现
# ---------------------------------------------------------------------------

def validate_generalization(
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
    val_env: TradingEnv,
    val_trajectory_states: np.ndarray,
    val_trajectory_actions: np.ndarray,
    val_trajectory_rewards: np.ndarray,
    device: torch.device,
    max_samples: int = 1024,
) -> Dict[str, Any]:
    """在验证集上评估 VQ 模型的泛化能力。

    训练集上 token accuracy 高不代表验证集上也好。如果验证集上
    token accuracy 显著下降，说明 VQ 模型过拟合了训练集的 DP 轨迹。

    如果没有预生成的验证集 DP 轨迹，可以用验证集环境的 horizon
    直接让 decoder 生成动作并在 env 中执行来评估。
    """
    encoder.eval()
    codebook.eval()
    decoder.eval()

    num_samples = min(len(val_trajectory_states), max_samples)
    indices = np.random.choice(len(val_trajectory_states), size=num_samples, replace=False)

    token_correct = 0
    token_total = 0
    exact_match = 0
    code_counts = np.zeros(codebook.num_codes, dtype=np.int64)

    batch_size = 256
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_idx = indices[start:end]

        s = torch.tensor(val_trajectory_states[batch_idx], dtype=torch.float32, device=device)
        a = torch.tensor(val_trajectory_actions[batch_idx], dtype=torch.long, device=device)
        r = torch.tensor(val_trajectory_rewards[batch_idx], dtype=torch.float32, device=device)

        with torch.no_grad():
            z_e = encoder(s, a, r)
            z_q_st, idx, _ = codebook.quantize(z_e)
            logits = decoder(s, z_q_st)
            preds = torch.argmax(logits, dim=-1)

        token_correct += int((preds == a).sum().item())
        token_total += int(a.numel())
        exact_match += int(torch.all(preds == a, dim=1).sum().item())
        code_counts += np.bincount(idx.cpu().numpy(), minlength=codebook.num_codes)

    token_accuracy = float(token_correct / max(token_total, 1))
    exact_match_rate = float(exact_match / max(num_samples, 1))
    used_codes = int(np.sum(code_counts > 0))

    return {
        "num_samples": num_samples,
        "val_token_accuracy": token_accuracy,
        "val_exact_match_rate": exact_match_rate,
        "val_used_codes": used_codes,
        "val_code_counts": {str(i): int(v) for i, v in enumerate(code_counts)},
        "warnings": [],
    }


def validate_val_env_returns(
    codebook: VQCodebook,
    decoder: VQDecoder,
    val_env: TradingEnv,
    device: torch.device,
    max_horizons: int = 256,
    normalizer: TrajectoryDataset | None = None,
) -> Dict[str, Any]:
    """在验证集环境中评估每个 archetype 的 return（无需 DP 轨迹）。

    对验证集的每个 horizon，用所有 K 个 archetype 分别生成动作并执行，
    统计每个 archetype 的 return 分布和最优 archetype 的 oracle return。
    """
    codebook.eval()
    decoder.eval()

    K = codebook.num_codes
    h = val_env.horizon
    num_horizons = min(val_env.num_horizons, max_horizons)

    per_archetype_returns = {k: [] for k in range(K)}
    oracle_returns = []
    best_archetype_counts = np.zeros(K, dtype=np.int64)

    for h_idx in range(num_horizons):
        start = h_idx * h
        end = min(start + h, len(val_env.states))
        horizon_states = val_env.states[start:end]

        horizon_returns = []
        for k in range(K):
            z_q = codebook.embeddings.weight[k].unsqueeze(0)
            actions = _decode_horizon(
                decoder, z_q, horizon_states, device,
                normalizer=normalizer,
            )
            result = _run_actions_in_env(val_env, h_idx, actions)
            per_archetype_returns[k].append(result["total_return"])
            horizon_returns.append(result["total_return"])

        best_k = int(np.argmax(horizon_returns))
        oracle_returns.append(horizon_returns[best_k])
        best_archetype_counts[best_k] += 1

    oracle_arr = np.array(oracle_returns, dtype=np.float64)
    per_archetype_summary = {}
    for k in range(K):
        arr = np.array(per_archetype_returns[k], dtype=np.float64)
        per_archetype_summary[str(k)] = {
            "return_mean": _safe_mean(arr),
            "return_std": _safe_std(arr),
            "positive_ratio": float(np.mean(arr > 0)) if arr.size > 0 else 0.0,
        }

    return {
        "num_horizons": num_horizons,
        "oracle_return_mean": _safe_mean(oracle_arr),
        "oracle_return_std": _safe_std(oracle_arr),
        "oracle_positive_ratio": float(np.mean(oracle_arr > 0)) if oracle_arr.size > 0 else 0.0,
        "best_archetype_distribution": {str(i): int(v) for i, v in enumerate(best_archetype_counts)},
        "per_archetype": per_archetype_summary,
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# 4. Decoder 动作分布偏移分析
# ---------------------------------------------------------------------------

def validate_decoder_action_shift(
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
    trajectory_dataset: TrajectoryDataset,
    device: torch.device,
    max_samples: int = 2048,
) -> Dict[str, Any]:
    """分析 decoder 重建动作相对于 DP 原始动作的系统性偏移。

    token accuracy 是全局指标，可能掩盖关键时刻的错误。
    本函数分析：
    - 在 DP 动作发生变化的关键步（change point）上的重建准确率
    - 各动作类别的 precision/recall 不平衡
    - decoder 是否系统性偏向某个动作（如过度预测 flat）
    """
    encoder.eval()
    codebook.eval()
    decoder.eval()

    num_samples = min(len(trajectory_dataset), max_samples)
    indices = np.random.choice(len(trajectory_dataset), size=num_samples, replace=False)

    # 统计
    change_point_correct = 0
    change_point_total = 0
    non_change_correct = 0
    non_change_total = 0
    dp_action_counts = np.zeros(3, dtype=np.int64)
    dec_action_counts = np.zeros(3, dtype=np.int64)
    # 在 change point 上的混淆矩阵
    change_confusion = np.zeros((3, 3), dtype=np.int64)

    batch_size = 256
    for start in range(0, num_samples, batch_size):
        end_idx = min(start + batch_size, num_samples)
        batch_indices = indices[start:end_idx]

        states_list, actions_list, rewards_list = [], [], []
        for i in batch_indices:
            s, a, r = trajectory_dataset[int(i)]
            states_list.append(s)
            actions_list.append(a)
            rewards_list.append(r)

        s_t = torch.tensor(np.array(states_list), dtype=torch.float32, device=device)
        a_t = torch.tensor(np.array(actions_list), dtype=torch.long, device=device)
        r_t = torch.tensor(np.array(rewards_list), dtype=torch.float32, device=device)

        with torch.no_grad():
            z_e = encoder(s_t, a_t, r_t)
            z_q_st, _, _ = codebook.quantize(z_e)
            logits = decoder(s_t, z_q_st)
            preds = torch.argmax(logits, dim=-1)

        a_np = a_t.cpu().numpy()
        p_np = preds.cpu().numpy()

        for row in range(a_np.shape[0]):
            for t in range(a_np.shape[1]):
                dp_action_counts[a_np[row, t]] += 1
                dec_action_counts[p_np[row, t]] += 1

                is_change = (t > 0 and a_np[row, t] != a_np[row, t - 1])
                if is_change:
                    change_point_total += 1
                    change_confusion[a_np[row, t], p_np[row, t]] += 1
                    if a_np[row, t] == p_np[row, t]:
                        change_point_correct += 1
                else:
                    non_change_total += 1
                    if a_np[row, t] == p_np[row, t]:
                        non_change_correct += 1

    change_accuracy = float(change_point_correct / max(change_point_total, 1))
    non_change_accuracy = float(non_change_correct / max(non_change_total, 1))

    # 动作分布偏移
    dp_dist = dp_action_counts.astype(np.float64) / max(dp_action_counts.sum(), 1)
    dec_dist = dec_action_counts.astype(np.float64) / max(dec_action_counts.sum(), 1)
    action_labels = ["short", "flat", "long"]

    report = {
        "num_samples": num_samples,
        "change_point_accuracy": change_accuracy,
        "non_change_accuracy": non_change_accuracy,
        "change_point_total": change_point_total,
        "non_change_total": non_change_total,
        "dp_action_distribution": {action_labels[i]: float(dp_dist[i]) for i in range(3)},
        "decoder_action_distribution": {action_labels[i]: float(dec_dist[i]) for i in range(3)},
        "action_distribution_shift": {
            action_labels[i]: float(dec_dist[i] - dp_dist[i]) for i in range(3)
        },
        "change_point_confusion_matrix": change_confusion.tolist(),
        "warnings": [],
    }

    if change_accuracy < 0.5:
        report["warnings"].append(
            f"decoder 在动作变化点的准确率仅 {change_accuracy:.2%}，"
            "关键交易时刻的重建质量差，会严重影响 Phase 2 的 archetype 可用性"
        )
    max_shift = max(abs(dec_dist[i] - dp_dist[i]) for i in range(3))
    if max_shift > 0.1:
        report["warnings"].append(
            f"decoder 动作分布偏移过大 (max_shift={max_shift:.4f})，"
            "可能系统性偏向某个动作"
        )

    return report


# ---------------------------------------------------------------------------
# 5. 统一入口：Phase I 环境级验证
# ---------------------------------------------------------------------------

def run_phase1_env_validation(
    config: Any,
    pair: str,
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
    train_env: TradingEnv,
    trajectory_dataset: TrajectoryDataset,
    device: torch.device,
    val_env: TradingEnv | None = None,
) -> Dict[str, Any]:
    """Phase I 环境级验证统一入口。

    在 Phase I 训练完成后调用，输出一份完整的环境级诊断报告。
    该报告与现有 validation.py 的报告互补：
    - validation.py: VQ 重建精度 + codebook 几何健康度
    - env_validation.py: archetype 在真实环境中的可用性

    Args:
        config: 全局配置
        pair: 交易对
        encoder: 训练好的 VQ encoder
        codebook: 训练好的 codebook
        decoder: 训练好的 decoder
        train_env: 训练集环境
        trajectory_dataset: DP 轨迹数据集
        device: 计算设备
        val_env: 验证集环境（可选，提供时会额外做验证集评估）

    Returns:
        完整的环境级验证报告 dict
    """
    logger.info("=" * 50)
    logger.info("Phase I 环境级验证开始: pair=%s", pair)

    report: Dict[str, Any] = {
        "pair": pair,
        "archetype_env_returns": {},
        "archetype_diversity": {},
        "decoder_action_shift": {},
        "val_env_returns": {},
        "all_warnings": [],
    }

    # 1. Archetype 环境级 return
    logger.info("[1/4] 评估每个 archetype 的环境级 return 分布...")
    env_returns = validate_archetype_env_returns(
        encoder=encoder,
        codebook=codebook,
        decoder=decoder,
        env=train_env,
        trajectory_dataset=trajectory_dataset,
        device=device,
    )
    report["archetype_env_returns"] = env_returns
    report["all_warnings"].extend(env_returns.get("warnings", []))

    logger.info(
        "  DP return: mean=%.4f, Decoded return: mean=%.4f, Gap: mean=%.4f",
        env_returns["dp_return_mean"],
        env_returns["decoded_return_mean"],
        env_returns["return_gap_mean"],
    )
    logger.info(
        "  正收益 archetype 数量: %d/%d",
        env_returns["positive_archetype_count"],
        codebook.num_codes,
    )
    for k_str, stats in env_returns["per_archetype"].items():
        logger.info(
            "  Archetype %s: count=%d, return=%.4f±%.4f, positive_ratio=%.2f%%",
            k_str, stats["count"], stats["return_mean"], stats["return_std"],
            stats["positive_ratio"] * 100,
        )

    # 2. Archetype 行为差异性
    logger.info("[2/4] 分析 archetype 之间的行为差异性...")
    diversity = validate_archetype_diversity(
        codebook=codebook,
        decoder=decoder,
        env=train_env,
        device=device,
        normalizer=trajectory_dataset,
    )
    report["archetype_diversity"] = diversity
    report["all_warnings"].extend(diversity.get("warnings", []))

    logger.info(
        "  Pairwise action agreement: mean=%.4f, min=%.4f, max=%.4f",
        diversity["pairwise_action_agreement_mean"],
        diversity["pairwise_action_agreement_min"],
        diversity["pairwise_action_agreement_max"],
    )
    logger.info(
        "  Pairwise JS divergence: mean=%.4f",
        diversity["pairwise_js_divergence_mean"],
    )
    for k_str, dist in diversity["per_archetype_action_distribution"].items():
        logger.info(
            "  Archetype %s action dist: short=%.2f%%, flat=%.2f%%, long=%.2f%%",
            k_str, dist["short"] * 100, dist["flat"] * 100, dist["long"] * 100,
        )

    # 3. Decoder 动作分布偏移
    logger.info("[3/4] 分析 decoder 动作分布偏移...")
    action_shift = validate_decoder_action_shift(
        encoder=encoder,
        codebook=codebook,
        decoder=decoder,
        trajectory_dataset=trajectory_dataset,
        device=device,
    )
    report["decoder_action_shift"] = action_shift
    report["all_warnings"].extend(action_shift.get("warnings", []))

    logger.info(
        "  Change point accuracy: %.2f%% (%d points), Non-change accuracy: %.2f%%",
        action_shift["change_point_accuracy"] * 100,
        action_shift["change_point_total"],
        action_shift["non_change_accuracy"] * 100,
    )
    logger.info(
        "  DP action dist:      short=%.2f%%, flat=%.2f%%, long=%.2f%%",
        action_shift["dp_action_distribution"]["short"] * 100,
        action_shift["dp_action_distribution"]["flat"] * 100,
        action_shift["dp_action_distribution"]["long"] * 100,
    )
    logger.info(
        "  Decoder action dist: short=%.2f%%, flat=%.2f%%, long=%.2f%%",
        action_shift["decoder_action_distribution"]["short"] * 100,
        action_shift["decoder_action_distribution"]["flat"] * 100,
        action_shift["decoder_action_distribution"]["long"] * 100,
    )

    # 4. 验证集环境 return（如果提供了 val_env）
    if val_env is not None and val_env.num_horizons > 0:
        logger.info("[4/4] 评估验证集上每个 archetype 的 return...")
        val_returns = validate_val_env_returns(
            codebook=codebook,
            decoder=decoder,
            val_env=val_env,
            device=device,
            normalizer=trajectory_dataset,
        )
        report["val_env_returns"] = val_returns
        report["all_warnings"].extend(val_returns.get("warnings", []))

        logger.info(
            "  验证集 oracle return: mean=%.4f, positive_ratio=%.2f%%",
            val_returns["oracle_return_mean"],
            val_returns["oracle_positive_ratio"] * 100,
        )
        logger.info(
            "  验证集 best archetype 分布: %s",
            val_returns["best_archetype_distribution"],
        )
        for k_str, stats in val_returns["per_archetype"].items():
            logger.info(
                "  Val Archetype %s: return=%.4f±%.4f, positive_ratio=%.2f%%",
                k_str, stats["return_mean"], stats["return_std"],
                stats["positive_ratio"] * 100,
            )
    else:
        logger.info("[4/4] 未提供验证集环境，跳过验证集评估")

    # 汇总 warnings
    if report["all_warnings"]:
        logger.warning("=" * 50)
        logger.warning("Phase I 环境级验证发现 %d 个警告:", len(report["all_warnings"]))
        for i, w in enumerate(report["all_warnings"], 1):
            logger.warning("  [%d] %s", i, w)
    else:
        logger.info("Phase I 环境级验证未发现警告")

    logger.info("Phase I 环境级验证完成")
    logger.info("=" * 50)

    return report
