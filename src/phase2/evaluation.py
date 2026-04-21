"""评估模块

功能说明:
    在验证集和训练子集上评估 SelectionAgent 的性能，
    提供详细的诊断信息。

论文相关:
    对应 Section 4.2 的 inference 过程。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.env.trading_env import TradingEnv
from src.phase1.codebook import VQCodebook
from src.phase1.vq_decoder import VQDecoder
from src.phase1.vq_encoder import VQEncoder
from src.phase2.selection_agent import SelectionAgent


def phase2_health_status(metrics: dict[str, Any]) -> tuple[str, str]:
    """给出 Phase II 验证结果的健康度标签与说明。"""
    avg_return = float(metrics.get("avg_return", 0.0))
    avg_cost = float(metrics.get("avg_execution_cost", 0.0))

    if avg_return < 0.0:
        return (
            "bad_negative_return",
            "验证集平均 return 为负，说明第二阶段策略当前方向存在明显问题；直接进入第三阶段通常难以彻底修复。",
        )

    # "微弱收益"判定: 收益仅与执行成本同量级，边际非常薄
    weak_threshold = max(1e-8, abs(avg_cost))
    if avg_return <= weak_threshold:
        return (
            "weak_edge",
            "验证集收益仅与执行成本同量级，属于微弱优势；建议先继续打磨第二阶段再进入第三阶段。",
        )

    return ("healthy", "验证集收益明显高于执行成本，第二阶段模型整体健康。")


def evaluate_on_validation(
    agent: SelectionAgent,
    codebook: VQCodebook,
    decoder: VQDecoder,
    val_env: TradingEnv,
    device: torch.device,
    max_horizons: int | None = None,
    # 需要从 train_phase2.py 导入的辅助函数
    get_horizon_start_states_fn=None,
    batch_decode_actions_fn=None,
    vectorized_execute_horizons_fn=None,
    aggregate_execution_diagnostics_fn=None,
    format_histogram_from_counts_fn=None,
    histogram_counts_fn=None,
) -> dict[str, Any]:
    """在验证集上评估 SelectionAgent，返回平均 horizon return 和诊断指标。

    # 需求 5.7: 定期在验证集上评估性能

    功能说明:
        在验证阶段，对每个 horizon 取首 bar 状态，使用当前策略贪心选择最优
        archetype（argmax），再通过 frozen decoder 执行整段微动作并累加收益。

        除平均 return 外，还额外输出 gross pnl / execution cost / turnover /
        direct flips / archetype histogram 等诊断项，便于区分"方向错"和"成本过高"。

    性能优化: 使用 batch_decode_actions + vectorized_execute_horizons
    替代逐 horizon 的 Python 循环。

    论文相关:
        - 对应 Section 4.2 的 inference 过程；
        - 训练时可以采样以保持探索，验证时通常用 argmax 检查 selector
          当前学到的 archetype 匹配能力；
        - 返回值仍然围绕 horizon-level 回报，与论文中的 r_sel 定义一致。

    Args:
        agent: SelectionAgent
        codebook: 冻结的码本
        decoder: 冻结的 Decoder
        val_env: 验证集环境
        device: 计算设备
        max_horizons: 若指定，则只评估前若干个 horizon，用于加速验证
        get_horizon_start_states_fn: 辅助函数
        batch_decode_actions_fn: 辅助函数
        vectorized_execute_horizons_fn: 辅助函数
        aggregate_execution_diagnostics_fn: 辅助函数
        format_histogram_from_counts_fn: 辅助函数
        histogram_counts_fn: 辅助函数

    Returns:
        dict[str, Any]: 平均 return 及执行诊断
    """
    agent.eval()
    num_horizons = val_env.num_horizons

    if num_horizons == 0:
        agent.train()
        return {"avg_return": 0.0, "selected_histogram": "[]"}

    if max_horizons is not None:
        num_horizons = min(num_horizons, int(max_horizons))

    horizon_indices = np.arange(num_horizons, dtype=np.int64)

    # 批量获取所有 horizon 起始状态
    states_np = get_horizon_start_states_fn(val_env, horizon_indices)
    states_t = torch.tensor(states_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        action_probs, _ = agent(states_t)
        selected_archetypes = torch.argmax(action_probs, dim=-1).detach().cpu().numpy()

    # 批量 decoder + 向量化执行
    archetype_t = torch.tensor(selected_archetypes, dtype=torch.long, device=device)
    all_actions_np = batch_decode_actions_fn(
        decoder=decoder,
        codebook=codebook,
        env=val_env,
        horizon_indices=horizon_indices,
        archetype_indices=archetype_t,
        device=device,
    )

    _, horizon_details = vectorized_execute_horizons_fn(
        env=val_env,
        horizon_indices=horizon_indices,
        all_actions=all_actions_np,
        need_diagnostics=True,
    )

    metrics = aggregate_execution_diagnostics_fn(horizon_details)
    metrics["selected_histogram"] = format_histogram_from_counts_fn(
        histogram_counts_fn(selected_archetypes, codebook.embeddings.weight.size(0))
    )
    metrics["avg_return"] = metrics.pop("avg_return")

    agent.train()
    return metrics


def evaluate_training_subset_diagnostics(
    agent: SelectionAgent,
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
    train_env: TradingEnv,
    demo_states: np.ndarray,
    demo_actions: np.ndarray,
    demo_rewards: np.ndarray,
    diagnostic_horizons: int,
    device: torch.device,
    # 需要从 train_phase2.py 导入的辅助函数
    get_horizon_start_states_fn=None,
    get_ground_truth_labels_fn=None,
    batch_decode_actions_fn=None,
    vectorized_execute_horizons_fn=None,
    aggregate_execution_diagnostics_fn=None,
    format_histogram_from_counts_fn=None,
    histogram_counts_fn=None,
) -> dict[str, Any]:
    """在训练子集上做 learned / random / oracle / fixed baseline 对照。

    功能说明:
        该诊断不直接参与训练，只用于定位负收益来源：
        - learned selector 是否优于 random；
        - gt oracle 是否明显高于 learned；
        - best fixed archetype 是否已经为负。

    性能优化:
        原实现对 learned / random / oracle / K 个 fixed 分别调用 _run_policy_on_horizons，
        共 K+3 次 batch_decode_actions + vectorized_execute_horizons。
        改为把所有策略的 actions 堆叠成 ((K+3)×subset_size,) 的大批量，
        一次 batch_decode_actions + 一次 vectorized_execute_horizons 完成，
        decoder 前向和 LOB slippage 预提取各只做一次。

    论文相关:
        这一步并不改变论文算法本身，而是对 Section 4.2 的 archetype selector
        做工程诊断，帮助判断瓶颈在 selector 还是在 frozen archetype 基座。
    """
    K = codebook.embeddings.weight.size(0)  # archetype 数量

    subset_size = min(int(diagnostic_horizons), train_env.num_horizons)
    if subset_size <= 0:
        return {
            "num_horizons": 0,
            "learned_return": 0.0,
            "random_return": 0.0,
            "oracle_return": 0.0,
            "best_fixed_return": 0.0,
            "best_fixed_idx": -1,
            "learned_gt_agreement": 0.0,
            "fixed_returns": "[]",
        }

    horizon_indices = np.random.choice(train_env.num_horizons, size=subset_size, replace=False)
    horizon_indices = np.asarray(horizon_indices, dtype=np.int64)

    # --- 计算各策略的 archetype 选择 ---
    states_np = get_horizon_start_states_fn(train_env, horizon_indices)
    states_t = torch.tensor(states_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        action_probs, _ = agent(states_t)
        learned_actions = torch.argmax(action_probs, dim=-1).detach().cpu().numpy()

    gt_labels = get_ground_truth_labels_fn(
        encoder=encoder, codebook=codebook,
        demo_states=demo_states, demo_actions=demo_actions, demo_rewards=demo_rewards,
        horizon_indices=horizon_indices, device=device,
    ).detach().cpu().numpy()

    rng = np.random.default_rng(12345)
    random_actions = rng.integers(0, K, size=subset_size, dtype=np.int64)

    # fixed_actions[k] = 全部选第 k 个 archetype，shape (K, subset_size)
    fixed_actions_all = np.stack(
        [np.full(subset_size, k, dtype=np.int64) for k in range(K)], axis=0
    )  # (K, subset_size)

    # --- 把所有策略的 actions 拼成一个大批量，一次 decode + execute ---
    # 顺序: [learned, random, oracle, fixed_0, fixed_1, ..., fixed_{K-1}]
    # 每段长度均为 subset_size，总长度 = (K+3) × subset_size
    all_archetypes = np.concatenate(
        [learned_actions, random_actions, gt_labels] + list(fixed_actions_all),
        axis=0,
    )  # ((K+3) × subset_size,)

    tiled_horizon_indices = np.tile(horizon_indices, K + 3)  # ((K+3) × subset_size,)

    archetype_t = torch.tensor(all_archetypes, dtype=torch.long, device=device)

    # 一次 batch_decode_actions：decoder 前向只跑一次
    all_actions_np = batch_decode_actions_fn(
        decoder=decoder, codebook=codebook, env=train_env,
        horizon_indices=tiled_horizon_indices,
        archetype_indices=archetype_t,
        device=device,
    )  # ((K+3)×subset_size, h)

    # 一次 vectorized_execute_horizons：LOB 预提取只做一次
    horizon_returns_np, horizon_details = vectorized_execute_horizons_fn(
        env=train_env,
        horizon_indices=tiled_horizon_indices,
        all_actions=all_actions_np,
        need_diagnostics=True,
    )  # ((K+3)×subset_size,)

    # --- 按段切分结果 ---
    def _slice_metrics(start: int) -> dict[str, Any]:
        """取第 start 段（长度 subset_size）的诊断指标。"""
        seg = slice(start * subset_size, (start + 1) * subset_size)
        return aggregate_execution_diagnostics_fn(horizon_details[seg])

    learned_metrics = _slice_metrics(0)
    random_metrics  = _slice_metrics(1)
    oracle_metrics  = _slice_metrics(2)

    fixed_returns: list[float] = []
    for k in range(K):
        fixed_returns.append(float(_slice_metrics(3 + k)["avg_return"]))

    best_fixed_idx = int(np.argmax(fixed_returns)) if fixed_returns else -1

    # selected_histogram 需要单独计算（aggregate_execution_diagnostics 不含）
    learned_metrics["selected_histogram"] = format_histogram_from_counts_fn(
        histogram_counts_fn(learned_actions, K)
    )

    return {
        "num_horizons": subset_size,
        "learned_return": float(learned_metrics["avg_return"]),
        "random_return": float(random_metrics["avg_return"]),
        "oracle_return": float(oracle_metrics["avg_return"]),
        "best_fixed_return": float(max(fixed_returns)) if fixed_returns else 0.0,
        "best_fixed_idx": best_fixed_idx,
        "learned_gt_agreement": float(np.mean(learned_actions == gt_labels)) if gt_labels.size > 0 else 0.0,
        "learned_selected_histogram": learned_metrics["selected_histogram"],
        "oracle_label_histogram": format_histogram_from_counts_fn(
            histogram_counts_fn(gt_labels, K)
        ),
        "fixed_returns": "[" + ", ".join(f"{idx}:{ret:.4f}" for idx, ret in enumerate(fixed_returns)) + "]",
        "learned_avg_gross_pnl": float(learned_metrics["avg_gross_pnl"]),
        "learned_avg_cost": float(learned_metrics["avg_execution_cost"]),
        "learned_avg_turnover": float(learned_metrics["avg_turnover"]),
        "learned_avg_direct_flips": float(learned_metrics["avg_direct_flips"]),
    }
