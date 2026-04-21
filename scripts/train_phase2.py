#!/usr/bin/env python
"""Phase II 训练脚本 — 原型选择（PPO 风格）

# 需求: 7.2, 5.3, 5.4, 5.5, 5.7, 7.4, 7.5, 7.6, 7.7
#
# 流程:
# 1. 加载 Phase I 模型（码本 + 冻结 Decoder），检查文件存在性
# 2. 加载特征数据，初始化 TradingEnv（训练集 + 验证集）
# 3. 初始化 SelectionAgent（Actor-Critic backbone）
# 4. 训练若干步（horizon 级别 RL / PPO 风格）
#    - 每个 horizon: agent 选择原型 → 冻结 decoder 生成 micro actions → env 执行 → 计算 horizon return
#    - PPO 更新: clipped surrogate objective + value loss + entropy bonus
#    - imitation / KL 惩罚: α(step) × KL(â_sel || π_sel)
#    - 其中 â_sel 来自冻结的 VQ encoder + codebook，对应论文 Eq.(5) 的 ground-truth archetype label
# 5. 定期在验证集上评估，保存最优检查点
# 6. 保存模型到 result/phase2_archetype_selection/
#
# 用法:
#   python scripts/train_phase2.py --pair BTC
#   python scripts/train_phase2.py --pair ETH --phase2-total-steps 1000000 --lr 1e-4
#
# 论文对应（AAAI26_ArchetypeTrader_core.md）:
# - Section 4.2 Archetype Selection
# - 高层状态: horizon 首 bar 的市场状态 s_sel
# - 高层动作: archetype index a_sel ∈ {0, ..., K-1}
# - 高层奖励: 一个 horizon 内的 step reward 累加得到 r_sel
# - 目标函数: Eq.(5) 中“环境收益 + ground-truth archetype 一致性约束”
#
# 实现说明:
# - 本脚本尽量保留你原代码的日志、方法分块和论文注解；
# - 在训练器上，从“单步 Actor-Critic”升级为“horizon-level PPO 风格”；
# - 不追求完全标准 PPO，而是优先保持论文语义与原工程结构的一致性。
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import parse_args
from src.data.feature_pipeline import FeaturePipeline
from src.env.trading_env import TradingEnv
from src.phase1.codebook import VQCodebook
from src.phase1.vq_decoder import VQDecoder
from src.phase1.vq_encoder import VQEncoder
from src.phase2.checkpoint import save_checkpoint
from src.phase2.config import get_current_selection_alpha, get_phase2_hparams
from src.phase2.data_loader import (
    create_environments,
    load_demo_trajectories,
    load_feature_data,
)
from src.phase2.diagnostics import (
    aggregate_execution_diagnostics,
    format_histogram_from_counts,
    histogram_counts,
)
from src.phase2.evaluation import (
    evaluate_on_validation,
    evaluate_training_subset_diagnostics,
    phase2_health_status,
)
from src.phase2.model_loader import load_phase1_model
from src.phase2.rollout import (
    batch_decode_actions,
    get_ground_truth_labels,
    get_horizon_start_states,
    vectorized_execute_horizons,
)
from src.phase2.selection_agent import SelectionAgent
from src.phase2.utils import parameter_grad_norm, set_reproducibility_seed
from src.utils.gpu_guard import log_and_guard_gpu_memory, reset_gpu_peak_memory_stats
from src.utils.logger import get_logger
from src.utils.normalizer import StateNormalizer
from src.utils.progress import should_disable_tqdm

logger = get_logger(__name__)

def collect_rollout_batch(
    agent: SelectionAgent,
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
    train_env: TradingEnv,
    demo_states: np.ndarray,
    demo_actions: np.ndarray,
    demo_rewards: np.ndarray,
    batch_size: int,
    device: torch.device,
    need_diagnostics: bool = True,
) -> dict[str, Any]:
    """采集一批 horizon-level rollout，用于 PPO 更新。

    功能说明:
        从训练环境中随机采样一批 horizon：
        1. 取每个 horizon 的起始状态作为 selector 输入；
        2. 用当前策略采样 archetype；
        3. 对应 archetype embedding 经 frozen decoder 生成 micro actions；
        4. 在 env 中执行并获得 horizon return；
        5. 保存 PPO 所需的 states / actions / old_log_probs / returns / advantages；
        6. 同时保存 demonstration 侧的 gt_labels 用于 imitation 正则；
        7. 额外记录 reward 拆分项和 archetype 直方图，便于定位负收益来源。

    论文相关:
        这一步是对 Section 4.2 的工程展开：
        - 高层状态: s_sel = horizon 首 bar 状态；
        - 高层动作: a_sel = archetype index；
        - 高层奖励: r_sel = Σ step_reward over horizon；
        - ground-truth label: â_sel = VQ encoder + codebook(demo chunk)。

    性能优化:
        - batch decoder: 一次前向传播为所有 horizon 生成 micro actions，
          避免逐 horizon 调用 decoder 的 Python 循环开销；
        - vectorized env: 持仓映射/价差/佣金全部 NumPy 向量化，
          仅 LOB slippage 仍需逐行查 DataFrame；
        - need_diagnostics: 非日志步跳过诊断统计，减少不必要的计算。

    实现说明:
        advantage 这里采用简化的一步形式 advantage = return - value，
        更接近原始代码结构；虽然不是全量 GAE，但已经满足 PPO 风格更新所需。

    Args:
        need_diagnostics: 是否计算完整诊断统计。False 时跳过 histogram/agreement
            等开销较大的统计，仅保留 horizon_return 用于 PPO 更新。

    Returns:
        dict[str, Any]: PPO 更新所需的一批张量和 rollout 诊断信息。
    """
    # 随机采样一批训练 horizon。保持和原实现一致：horizon 是 Phase II 的基本决策单位。
    horizon_indices = np.random.randint(0, train_env.num_horizons, size=batch_size)

    # 获取 horizon 起始状态，对应论文中的 s_sel。
    states_np = get_horizon_start_states(train_env, horizon_indices)
    states_t = torch.tensor(states_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        # Section 4.2: Agent 选择原型
        # 返回所有原型的策略概率和价值函数输出
        action_probs, values = agent(states_t)
        greedy_actions = torch.argmax(action_probs, dim=-1)
        dist = torch.distributions.Categorical(probs=action_probs)

        # 从策略分布中采样原型索引
        actions = dist.sample()  # (batch,)
        old_log_probs = dist.log_prob(actions)  # (batch,)

        # Eq.(5): 获取 ground-truth archetype label â_sel
        # 使用冻结的 VQ encoder + codebook 对这些 horizon 的 DP 示范轨迹编码。
        gt_labels = get_ground_truth_labels(
            encoder,
            codebook,
            demo_states,
            demo_actions,
            demo_rewards,
            horizon_indices,
            device,
        )

    # ---- 批量 decoder 推理: 一次前向传播生成所有 horizon 的 micro actions ----
    all_actions_np = batch_decode_actions(
        decoder=decoder,
        codebook=codebook,
        env=train_env,
        horizon_indices=horizon_indices,
        archetype_indices=actions,
        device=device,
    )  # (batch_size, h)

    # ---- 向量化 env 执行: NumPy 批量计算 reward ----
    horizon_returns_np, rollout_details = vectorized_execute_horizons(
        env=train_env,
        horizon_indices=horizon_indices,
        all_actions=all_actions_np,
        need_diagnostics=need_diagnostics,
    )

    raw_returns_t = torch.tensor(horizon_returns_np, dtype=torch.float32, device=device)
    returns_t = raw_returns_t.clone()

    # Return 归一化：将 horizon return 标准化到零均值单位方差，
    # 避免 value loss 因 return 绝对值过大（ETH 持仓 100 × 72 步）而爆炸，
    # 从而防止 critic 梯度淹没 actor 梯度导致策略坍缩。
    if batch_size > 1:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std(unbiased=False) + 1e-8)

    values_t = values.squeeze(-1).detach()

    # Section 4.2 / PPO 版本: advantage = R - V(s)
    # 这里保持与原代码同一语义，只是改成 batch 形式。
    raw_advantages_t = returns_t - values_t
    advantages_t = raw_advantages_t.clone()

    # PPO 中通常会做 advantage normalization，以稳定比例项 ratio 的更新。
    if batch_size > 1:
        advantages_t = (
            advantages_t - advantages_t.mean()
        ) / (advantages_t.std(unbiased=False) + 1e-8)

    # ---- 诊断统计: 仅在 need_diagnostics=True 时计算完整指标 ----
    if need_diagnostics:
        actions_np = actions.detach().cpu().numpy()
        greedy_np = greedy_actions.detach().cpu().numpy()
        gt_np = gt_labels.detach().cpu().numpy()

        diagnostics = aggregate_execution_diagnostics(rollout_details)
        diagnostics.update(
            {
                "raw_adv_mean": float(raw_advantages_t.mean().item()) if raw_advantages_t.numel() > 0 else 0.0,
                "raw_adv_std": float(raw_advantages_t.std(unbiased=False).item()) if raw_advantages_t.numel() > 0 else 0.0,
                "sampled_archetype_histogram": format_histogram_from_counts(
                    histogram_counts(actions_np, agent.num_archetypes)
                ),
                "greedy_archetype_histogram": format_histogram_from_counts(
                    histogram_counts(greedy_np, agent.num_archetypes)
                ),
                "gt_label_histogram": format_histogram_from_counts(
                    histogram_counts(gt_np, agent.num_archetypes)
                ),
                # gt_agree: selector 与 VQ encoder label 的一致性（诊断用）
                "sampled_gt_agreement": float(np.mean(actions_np == gt_np)) if gt_np.size > 0 else 0.0,
                "greedy_gt_agreement": float(np.mean(greedy_np == gt_np)) if gt_np.size > 0 else 0.0,
            }
        )
    else:
        diagnostics = {
            "avg_return": float(horizon_returns_np.mean()),
        }

    return {
        "states": states_t,
        "actions": actions.detach(),
        "old_log_probs": old_log_probs.detach(),
        "raw_returns": raw_returns_t.detach(),
        "returns": returns_t.detach(),
        "advantages": advantages_t.detach(),
        "gt_labels": gt_labels.detach(),
        "diagnostics": diagnostics,
    }


def ppo_update(
    agent: SelectionAgent,
    optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    alpha: float,
    imitation_min_raw_return: float,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    ppo_epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
    device: torch.device,
) -> dict[str, float]:
    """对同一批 rollout 执行多轮 PPO 更新。

    功能说明:
        对 collect_rollout_batch 收集到的 on-policy 数据进行多轮 minibatch 更新，
        这是把原始“单样本即时更新”改成“PPO 风格批量更新”的核心函数。

    论文相关:
        - 论文 Eq.(5) 给出了“环境奖励 + archetype 一致性约束”的目标；
        - 这里把该目标分解为四项：
          1) PPO clipped policy loss：负责优化 selector 的策略改进；
          2) value loss：估计 horizon return；
          3) entropy bonus：维持 archetype 探索；
          4) imitation loss：实现 KL(â_sel || π_sel) 的 one-hot 等价形式。

    实现说明:
        - imitation_loss 使用 F.nll_loss(log(action_probs), gt_labels)；
        - 对于 one-hot 的 â_sel，这与 KL(one_hot || π) 只差常数项，
          因此可视为论文 Eq.(5) 中 KL regularization 的稳定实现。

    Returns:
        dict[str, float]: 本轮 PPO 更新的统计量，供日志打印与调试。
    """
    states = batch["states"]
    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"]
    raw_returns = batch["raw_returns"]
    returns = batch["returns"]
    advantages = batch["advantages"]
    gt_labels = batch["gt_labels"]

    batch_size = states.size(0)
    minibatch_size = min(minibatch_size, batch_size)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    imitation_losses: list[float] = []
    entropies: list[float] = []
    total_losses: list[float] = []
    clip_fractions: list[float] = []
    approx_kls: list[float] = []
    imitation_mask_fractions: list[float] = []
    policy_grad_norms: list[float] = []
    value_grad_norms: list[float] = []
    shared_grad_norms: list[float] = []

    for _ in range(ppo_epochs):
        perm = torch.randperm(batch_size, device=device)

        for start in range(0, batch_size, minibatch_size):
            idx = perm[start : start + minibatch_size]

            mb_states = states[idx]
            mb_actions = actions[idx]
            mb_old_log_probs = old_log_probs[idx]
            mb_raw_returns = raw_returns[idx]
            mb_returns = returns[idx]
            mb_advantages = advantages[idx]
            mb_gt_labels = gt_labels[idx]

            # 重新计算当前 policy 下的概率分布和 value，构造 PPO ratio。
            action_probs, values = agent(mb_states)
            dist = torch.distributions.Categorical(probs=action_probs)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            # PPO clipped surrogate objective。
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surrogate1 = ratio * mb_advantages
            surrogate2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            # Value loss = (R - V(s))^2；保持和原始实现同一含义。
            value_pred = values.squeeze(-1)
            value_loss = F.mse_loss(value_pred, mb_returns)

            # Eq.(5): KL(â_sel || π_sel)。
            # 只对 raw horizon return 超过阈值的样本施加 imitation，
            # 避免把 policy 拉向“只是标准化后优势为正、但绝对收益很薄”的 archetype。
            pos_mask = (mb_raw_returns > imitation_min_raw_return).float()  # (mb,)
            per_sample_nll = F.nll_loss(
                torch.log(action_probs + 1e-8), mb_gt_labels, reduction="none"
            )  # (mb,)
            imitation_loss = (pos_mask * per_sample_nll).sum() / (pos_mask.sum() + 1e-8)

            # Actor 损失：PPO policy + entropy + imitation prior。
            # Critic 损失单独优化，避免 value loss 梯度劫持 shared backbone。
            actor_loss = (
                policy_loss
                - ent_coef * entropy
                + alpha * imitation_loss
            )
            critic_loss = vf_coef * value_loss

            # 总损失仅用于日志记录。
            total_loss = actor_loss + critic_loss

            # 先计算两路梯度，再统一 step，避免 in-place 修改参数后
            # 第二次 backward 遇到 stale tensor version 的问题。
            optimizer.zero_grad()
            critic_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            critic_loss.backward()

            policy_grad_norm = parameter_grad_norm(agent.policy_head.parameters())
            shared_grad_norm = parameter_grad_norm(agent.shared.parameters())
            value_grad_norm = parameter_grad_norm(agent.value_head.parameters())

            torch.nn.utils.clip_grad_norm_(
                list(agent.shared.parameters()) + list(agent.policy_head.parameters()),
                max_grad_norm,
            )
            torch.nn.utils.clip_grad_norm_(agent.value_head.parameters(), max_grad_norm)

            optimizer.step()
            critic_optimizer.step()

            clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()
            approx_kl = (mb_old_log_probs - new_log_probs).mean()

            policy_losses.append(float(policy_loss.detach().item()))
            value_losses.append(float(value_loss.detach().item()))
            imitation_losses.append(float(imitation_loss.detach().item()))
            entropies.append(float(entropy.detach().item()))
            total_losses.append(float(total_loss.detach().item()))
            clip_fractions.append(float(clip_fraction.detach().item()))
            approx_kls.append(float(approx_kl.detach().item()))
            imitation_mask_fractions.append(float(pos_mask.mean().detach().item()))
            policy_grad_norms.append(float(policy_grad_norm))
            value_grad_norms.append(float(value_grad_norm))
            shared_grad_norms.append(float(shared_grad_norm))

    return {
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "imitation_loss": float(np.mean(imitation_losses)) if imitation_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
        "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        "imitation_mask_fraction": float(np.mean(imitation_mask_fractions)) if imitation_mask_fractions else 0.0,
        "policy_grad_norm": float(np.mean(policy_grad_norms)) if policy_grad_norms else 0.0,
        "value_grad_norm": float(np.mean(value_grad_norms)) if value_grad_norms else 0.0,
        "shared_grad_norm": float(np.mean(shared_grad_norms)) if shared_grad_norms else 0.0,
    }

def run_training_loop(
    agent: SelectionAgent,
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
    eval_agent: SelectionAgent,
    eval_encoder: VQEncoder,
    eval_codebook: VQCodebook,
    eval_decoder: VQDecoder,
    eval_device: torch.device,
    train_env: TradingEnv,
    val_env: TradingEnv,
    demo_states: np.ndarray,
    demo_actions: np.ndarray,
    demo_rewards: np.ndarray,
    optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    alpha: float,
    total_steps: int,
    val_interval: int,
    log_interval: int,
    save_path: str,
    config: Any,
    ppo_hparams: dict[str, Any],
    device: torch.device,
) -> tuple[float, list[float], int]:
    """执行 Phase II 训练循环（PPO 版本）。

    Args:
        agent: SelectionAgent
        encoder: 冻结的 VQEncoder
        codebook: 冻结的 VQCodebook
        decoder: 冻结的 VQDecoder
        train_env: 训练集环境
        val_env: 验证集环境
        demo_states: DP 示范轨迹状态 (N, h, state_dim)
        demo_actions: DP 示范轨迹动作 (N, h)
        demo_rewards: DP 示范轨迹奖励 (N, h)
        optimizer: 优化器
        alpha: KL / imitation 惩罚系数
        total_steps: 总训练步数（以 horizon 样本数计）
        val_interval: 验证间隔
        log_interval: 日志间隔
        save_path: 最优模型保存路径
        config: 配置对象
        ppo_hparams: PPO 相关配置字典
        device: 计算设备

    功能说明:
        这是 Phase II 的主训练入口：
        反复执行“收集一批 horizon rollout → 多轮 PPO 更新 → 周期性验证与保存”。

        相比原版本，新增了三类诊断：
        1) rollout 奖励拆分（gross pnl / cost / turnover / flips）；
        2) actor/critic 梯度与 approx_kl；
        3) 训练子集上的 learned / random / oracle / fixed baseline 对照。

    论文相关:
        - 对应 Section 4.2 的 horizon-level RL；
        - 目标函数核心仍然来自 Eq.(5)：
          J = E[Σ γ^t r_sel - α × KL(â_sel || π_sel)]；
        - 这里把原先的单步 Actor-Critic 训练器升级为 PPO 风格，
          但高层状态/动作/奖励和 demonstration archetype regularization 均保持不变。

    Returns:
        best_val_return: 最优验证集 return
        reward_history: 奖励历史
        step_count: 实际训练步数
    """
    best_val_return = float("-inf")
    reward_history: list[float] = []
    step_count = 0
    last_stats: dict[str, float] = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "imitation_loss": 0.0,
        "imitation_mask_fraction": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
        "clip_fraction": 0.0,
        "approx_kl": 0.0,
        "policy_grad_norm": 0.0,
        "value_grad_norm": 0.0,
        "shared_grad_norm": 0.0,
    }
    last_batch_diag: dict[str, Any] = {
        "avg_return": 0.0,
        "avg_gross_pnl": 0.0,
        "avg_execution_cost": 0.0,
        "avg_turnover": 0.0,
        "avg_direct_flips": 0.0,
        "sampled_gt_agreement": 0.0,
        "greedy_gt_agreement": 0.0,
        "sampled_archetype_histogram": "[]",
        "gt_label_histogram": "[]",
        "decoder_action_histogram": "[]",
        "raw_adv_mean": 0.0,
        "raw_adv_std": 0.0,
    }

    rollout_batch_size = int(ppo_hparams["rollout_batch_size"])
    ppo_epochs = int(ppo_hparams["ppo_epochs"])
    minibatch_size = int(ppo_hparams["minibatch_size"])
    clip_eps = float(ppo_hparams["clip_eps"])
    vf_coef = float(ppo_hparams["vf_coef"])
    ent_coef = float(ppo_hparams["ent_coef"])
    max_grad_norm = float(ppo_hparams["max_grad_norm"])
    eval_max_horizons = ppo_hparams["eval_max_horizons"]
    diagnostic_horizons = int(ppo_hparams["diagnostic_horizons"])
    alpha_schedule = str(ppo_hparams["alpha_schedule"])
    alpha_final_ratio = float(ppo_hparams["alpha_final_ratio"])
    imitation_min_raw_return = float(ppo_hparams["imitation_min_raw_return"])

    next_log_step = log_interval
    next_val_step = val_interval

    # 保留原有日志。
    logger.info("开始训练: %d 步", total_steps)
    # 新增 PPO 训练器细节日志，便于和单步 Actor-Critic 区分。
    logger.info(
        "开始训练: total_steps=%d, rollout_batch=%d, ppo_epochs=%d, minibatch=%d, clip_eps=%.3f",
        total_steps,
        rollout_batch_size,
        ppo_epochs,
        minibatch_size,
        clip_eps,
    )

    disable_tqdm = should_disable_tqdm()
    pbar = tqdm(
        total=total_steps,
        desc="Phase II 训练",
        unit="step",
        dynamic_ncols=True,
        disable=disable_tqdm,
    )
    while step_count < total_steps:
        current_batch_size = min(rollout_batch_size, total_steps - step_count)

        # 降低诊断频率: 仅在即将输出日志、首批、或最后一批时计算完整诊断，
        # 其余步骤跳过 histogram/agreement 等开销较大的统计。
        is_first_batch = (step_count == 0)
        is_log_step = (step_count + current_batch_size >= next_log_step)
        is_val_step = (step_count + current_batch_size >= next_val_step)
        is_last_batch = (step_count + current_batch_size >= total_steps)
        need_diag = is_first_batch or is_log_step or is_val_step or is_last_batch

        batch = collect_rollout_batch(
            agent=agent,
            encoder=encoder,
            codebook=codebook,
            decoder=decoder,
            train_env=train_env,
            demo_states=demo_states,
            demo_actions=demo_actions,
            demo_rewards=demo_rewards,
            batch_size=current_batch_size,
            device=device,
            need_diagnostics=need_diag,
        )
        log_and_guard_gpu_memory(
            logger,
            stage=f"Phase II rollout_batch(step={step_count}, batch={current_batch_size})",
            device=device,
            force_log=need_diag,
        )
        if need_diag:
            last_batch_diag = batch["diagnostics"]

        if step_count == 0:
            current_alpha = get_current_selection_alpha(
                initial_alpha=alpha,
                schedule=alpha_schedule,
                final_ratio=alpha_final_ratio,
                step_count=step_count,
                total_steps=total_steps,
            )
            logger.info(
                "首批 rollout 形状: states=%s, actions=%s, returns=%s, advantages=%s, gt_labels=%s",
                tuple(batch["states"].shape),
                tuple(batch["actions"].shape),
                tuple(batch["returns"].shape),
                tuple(batch["advantages"].shape),
                tuple(batch["gt_labels"].shape),
            )
            logger.info(
                "首批 PPO / imitation 配置: alpha=%.4f, alpha_schedule=%s, alpha_final_ratio=%.4f, imitation_min_raw_return=%.4f",
                current_alpha,
                alpha_schedule,
                alpha_final_ratio,
                imitation_min_raw_return,
            )
            logger.info(
                "首批 rollout 诊断: gross=%.4f, cost=%.4f, turnover=%.4f, flips=%.4f, sampled_hist=%s, gt_hist=%s, sampled_agree=%.4f, greedy_agree=%.4f",
                last_batch_diag["avg_gross_pnl"],
                last_batch_diag["avg_execution_cost"],
                last_batch_diag["avg_turnover"],
                last_batch_diag["avg_direct_flips"],
                last_batch_diag["sampled_archetype_histogram"],
                last_batch_diag["gt_label_histogram"],
                last_batch_diag["sampled_gt_agreement"],
                last_batch_diag["greedy_gt_agreement"],
            )

        current_alpha = get_current_selection_alpha(
            initial_alpha=alpha,
            schedule=alpha_schedule,
            final_ratio=alpha_final_ratio,
            step_count=step_count,
            total_steps=total_steps,
        )
        last_stats = ppo_update(
            agent=agent,
            optimizer=optimizer,
            critic_optimizer=critic_optimizer,
            batch=batch,
            alpha=current_alpha,
            imitation_min_raw_return=imitation_min_raw_return,
            clip_eps=clip_eps,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            max_grad_norm=max_grad_norm,
            device=device,
        )
        log_and_guard_gpu_memory(
            logger,
            stage=f"Phase II ppo_update(step={step_count}, batch={current_batch_size})",
            device=device,
            force_log=need_diag,
        )

        batch_returns = batch["returns"].detach().cpu().tolist()
        reward_history.extend(float(x) for x in batch_returns)
        step_count += current_batch_size

        # 更新进度条
        recent_rewards = reward_history[-min(log_interval, len(reward_history)) :]
        avg_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        pbar.update(current_batch_size)
        pbar.set_postfix({
            "avg_r": f"{avg_reward:.4f}",
            "loss": f"{last_stats['total_loss']:.4f}",
            "policy": f"{last_stats['policy_loss']:.4f}",
            "kl": f"{last_stats['approx_kl']:.4f}",
            "alpha": f"{current_alpha:.3f}",
            "val": f"{best_val_return:.4f}",
        })

        # 日志输出
        if step_count >= next_log_step or step_count == total_steps:
            batch_avg_reward = float(np.mean(batch_returns)) if batch_returns else 0.0
            logger.info(
                "Step %7d/%d — avg_reward=%.4f, batch_reward=%.4f, alpha=%.4f, total=%.4f, policy=%.4f, value=%.4f, imitation=%.4f, imitation_mask=%.4f, entropy=%.4f, clipfrac=%.4f, kl=%.4f, p_gn=%.4f, v_gn=%.4f, shared_gn=%.4f",
                step_count,
                total_steps,
                avg_reward,
                batch_avg_reward,
                current_alpha,
                last_stats["total_loss"],
                last_stats["policy_loss"],
                last_stats["value_loss"],
                last_stats["imitation_loss"],
                last_stats["imitation_mask_fraction"],
                last_stats["entropy"],
                last_stats["clip_fraction"],
                last_stats["approx_kl"],
                last_stats["policy_grad_norm"],
                last_stats["value_grad_norm"],
                last_stats["shared_grad_norm"],
            )
            logger.info(
                "Step %7d/%d — rollout诊断: gross=%.4f, cost=%.4f (commission=%.4f, slippage=%.4f), turnover=%.4f, flips=%.4f, raw_adv_mean=%.4f, raw_adv_std=%.4f, sampled_agree=%.4f, greedy_agree=%.4f",
                step_count,
                total_steps,
                last_batch_diag["avg_gross_pnl"],
                last_batch_diag["avg_execution_cost"],
                last_batch_diag["avg_commission"],
                last_batch_diag["avg_slippage"],
                last_batch_diag["avg_turnover"],
                last_batch_diag["avg_direct_flips"],
                last_batch_diag["raw_adv_mean"],
                last_batch_diag["raw_adv_std"],
                last_batch_diag["sampled_gt_agreement"],
                last_batch_diag["greedy_gt_agreement"],
            )
            logger.info(
                "Step %7d/%d — archetype直方图: sampled=%s, greedy=%s, gt=%s, decoder_actions=%s",
                step_count,
                total_steps,
                last_batch_diag["sampled_archetype_histogram"],
                last_batch_diag["greedy_archetype_histogram"],
                last_batch_diag["gt_label_histogram"],
                last_batch_diag["decoder_action_histogram"],
            )
            next_log_step = ((step_count // log_interval) + 1) * log_interval

        # 需求 5.7: 定期在验证集上评估，保存最优检查点
        if step_count >= next_val_step or step_count == total_steps:
            pbar.set_description("验证集评估中")
            # 验证与诊断使用 CPU，避免占用 GPU 显存（16G 显卡也能稳定跑训练）。
            eval_agent.load_state_dict(agent.state_dict(), strict=True)
            val_metrics = evaluate_on_validation(
                agent=eval_agent,
                codebook=eval_codebook,
                decoder=eval_decoder,
                val_env=val_env,
                device=eval_device,
                max_horizons=eval_max_horizons,
            )
            val_return = float(val_metrics["avg_return"])
            logger.info(
                "验证集评估 (step %d): avg_return=%.4f (best=%.4f), gross=%.4f, cost=%.4f, turnover=%.4f, flips=%.4f, selected=%s",
                step_count,
                val_return,
                best_val_return,
                val_metrics["avg_gross_pnl"],
                val_metrics["avg_execution_cost"],
                val_metrics["avg_turnover"],
                val_metrics["avg_direct_flips"],
                val_metrics["selected_histogram"],
            )

            train_diag = evaluate_training_subset_diagnostics(
                agent=eval_agent,
                encoder=eval_encoder,
                codebook=eval_codebook,
                decoder=eval_decoder,
                train_env=train_env,
                demo_states=demo_states,
                demo_actions=demo_actions,
                demo_rewards=demo_rewards,
                diagnostic_horizons=diagnostic_horizons,
                device=eval_device,
            )
            logger.info(
                "训练子集诊断 (n=%d): learned=%.4f, random=%.4f, oracle=%.4f, best_fixed=%.4f(k=%d), gt_agree=%.4f",
                train_diag["num_horizons"],
                train_diag["learned_return"],
                train_diag["random_return"],
                train_diag["oracle_return"],
                train_diag["best_fixed_return"],
                train_diag["best_fixed_idx"],
                train_diag["learned_gt_agreement"],
            )
            logger.info(
                "训练子集诊断 (n=%d): gross=%.4f, cost=%.4f, turnover=%.4f, flips=%.4f, learned_hist=%s, oracle_hist=%s",
                train_diag["num_horizons"],
                train_diag["learned_avg_gross_pnl"],
                train_diag["learned_avg_cost"],
                train_diag["learned_avg_turnover"],
                train_diag["learned_avg_direct_flips"],
                train_diag["learned_selected_histogram"],
                train_diag["oracle_label_histogram"],
            )
            logger.info(
                "训练子集固定原型收益: %s",
                train_diag["fixed_returns"],
            )

            if val_return > best_val_return:
                best_val_return = val_return
                save_checkpoint(
                    save_path=save_path,
                    agent=agent,
                    optimizer=optimizer,
                    critic_optimizer=critic_optimizer,
                    reward_history=reward_history,
                    best_val_return=best_val_return,
                    step_count=step_count,
                    config=config,
                    ppo_hparams=ppo_hparams,
                )
                logger.info("最优模型已保存到 %s (val_return=%.4f)", save_path, val_return)

            pbar.set_description("Phase II 训练")
            pbar.set_postfix({
                "avg_r": f"{avg_reward:.4f}",
                "loss": f"{last_stats['total_loss']:.4f}",
                "policy": f"{last_stats['policy_loss']:.4f}",
                "kl": f"{last_stats['approx_kl']:.4f}",
                "alpha": f"{current_alpha:.3f}",
                "val": f"{best_val_return:.4f}",
            })
            next_val_step = ((step_count // val_interval) + 1) * val_interval

    pbar.close()

    return best_val_return, reward_history, step_count


def main() -> None:
    """Phase II 训练入口。

    功能说明:
        负责串联整个训练流程：解析配置、加载 Phase I 模型、准备训练/验证环境、
        加载 DP demonstration、初始化 SelectionAgent、执行 PPO 风格训练、
        并保存最优与最终模型。

    论文相关:
        - Step 1: 使用 Phase I 学到的 archetype discovery 结果；
        - Step 2~4: 对应 Phase II 的 archetype selection；
        - 训练目标基于 Eq.(5)，但优化器实现采用 horizon-level PPO 风格。
    """
    # ----------------------------------------------------------------
    # Step 0: 解析配置
    # ----------------------------------------------------------------
    config = parse_args()
    pair = config.pairs[0]  # 单交易对训练
    set_reproducibility_seed(config.phase1_sampling_seed)
    ppo_hparams = get_phase2_hparams(config)

    logger.info("Phase II 训练开始: pair=%s", pair)
    logger.info("随机种子: phase1_sampling_seed=%d（Phase I/II 对齐）", config.phase1_sampling_seed)
    logger.info(
        "超参数: total_steps=%d, lr=%.1e, selection_alpha=%.2f, alpha_schedule=%s, alpha_final_ratio=%.2f, num_archetypes=%d, discount_factor=%.2f",
        config.phase2_total_steps,
        config.learning_rate,
        config.selection_alpha,
        ppo_hparams["alpha_schedule"],
        ppo_hparams["alpha_final_ratio"],
        config.num_archetypes,
        config.discount_factor,
    )
    logger.info(
        "PPO 超参数: rollout_batch=%d, ppo_epochs=%d, minibatch=%d, clip_eps=%.3f, vf_coef=%.3f, ent_coef=%.4f, max_grad_norm=%.2f, val_interval_multiplier=%d, imitation_min_raw_return=%.4f, diagnostic_horizons=%d",
        ppo_hparams["rollout_batch_size"],
        ppo_hparams["ppo_epochs"],
        ppo_hparams["minibatch_size"],
        ppo_hparams["clip_eps"],
        ppo_hparams["vf_coef"],
        ppo_hparams["ent_coef"],
        ppo_hparams["max_grad_norm"],
        ppo_hparams["val_interval_multiplier"],
        ppo_hparams["imitation_min_raw_return"],
        ppo_hparams["diagnostic_horizons"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)
    logger.info("结果目录批次: %s", config.train_batch_id)
    reset_gpu_peak_memory_stats(device)
    log_and_guard_gpu_memory(logger, stage="Phase II startup", device=device, force_log=True)

    # ----------------------------------------------------------------
    # Step 1: 加载 Phase I 模型（编码器 + 码本 + 冻结 Decoder）
    # ----------------------------------------------------------------
    encoder, codebook, decoder, normalizer = load_phase1_model(config, pair, device)
    log_and_guard_gpu_memory(logger, stage="Phase II after Phase I model load", device=device, force_log=True)

    # ----------------------------------------------------------------
    # Step 2: 加载特征数据，初始化 TradingEnv
    # ----------------------------------------------------------------
    train_states, val_states, train_prices, val_prices, train_df, val_df = load_feature_data(
        config, pair, normalizer
    )

    train_env, val_env = create_environments(
        config, pair, train_states, val_states, train_prices, val_prices, train_df, val_df
    )
    if train_env.num_horizons == 0:
        logger.error("训练集 horizon 数量为 0，无法训练")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Step 2.5: 加载 DP 示范轨迹（用于 Eq.5 的 ground-truth archetype label）
    # DP 轨迹文件由 Phase I 的 DPPlanner.generate_trajectories() 生成，
    # 前 num_horizons 条与训练环境 horizon 索引 1:1 对齐。
    # ----------------------------------------------------------------
    demo_states, demo_actions, demo_rewards = load_demo_trajectories(config, pair, normalizer)

    if demo_states.shape[0] < train_env.num_horizons:
        raise ValueError(
            "DP 示范轨迹数量少于训练环境的 horizon 数量，无法为每个训练 horizon 提供 ground-truth archetype label。"
            f" demo={demo_states.shape[0]}, train_horizons={train_env.num_horizons}"
        )

    # ----------------------------------------------------------------
    # Step 3: 初始化 SelectionAgent
    # ----------------------------------------------------------------
    agent = SelectionAgent(
        state_dim=config.state_dim,
        num_archetypes=config.num_archetypes,
        hidden_dim=config.phase2_hidden_dim,
        bottleneck_dim=config.phase2_bottleneck_dim,
    ).to(device)

    # Phase II 验证/诊断统一走 CPU，显著降低验证阶段显存峰值（适配 16G 显卡）。
    eval_device = torch.device("cpu")
    eval_agent = SelectionAgent(
        state_dim=config.state_dim,
        num_archetypes=config.num_archetypes,
        hidden_dim=config.phase2_hidden_dim,
        bottleneck_dim=config.phase2_bottleneck_dim,
    ).to(eval_device)
    eval_encoder = VQEncoder(
        state_dim=encoder.state_dim,
        action_dim=encoder.action_dim,
        hidden_dim=encoder.hidden_dim,
        latent_dim=encoder.latent_dim,
    ).to(eval_device)
    eval_encoder.load_state_dict(encoder.state_dict(), strict=True)
    eval_codebook = VQCodebook(
        num_codes=codebook.num_codes,
        code_dim=codebook.code_dim,
    ).to(eval_device)
    eval_codebook.load_state_dict(codebook.state_dict(), strict=True)
    eval_decoder = VQDecoder(
        state_dim=decoder.state_dim,
        code_dim=decoder.code_dim,
        hidden_dim=decoder.hidden_dim,
        action_dim=decoder.action_dim,
    ).to(eval_device)
    eval_decoder.load_state_dict(decoder.state_dict(), strict=True)
    for param in eval_encoder.parameters():
        param.requires_grad = False
    for param in eval_codebook.parameters():
        param.requires_grad = False
    for param in eval_decoder.parameters():
        param.requires_grad = False
    eval_encoder.eval()
    eval_codebook.eval()
    eval_decoder.eval()
    logger.info("Phase II 验证设备: %s（训练设备: %s）", eval_device, device)

    logger.info(
        "SelectionAgent 初始化完成: params=%d",
        sum(p.numel() for p in agent.parameters()),
    )
    optimizer = torch.optim.Adam(
        list(agent.shared.parameters()) + list(agent.policy_head.parameters()),
        lr=config.learning_rate,
    )
    critic_optimizer = torch.optim.Adam(
        agent.value_head.parameters(),
        lr=config.learning_rate * 3,  # critic 用更高学习率独立拟合 value
    )

    # ----------------------------------------------------------------
    # Step 4: 训练循环 — horizon 级别 RL（PPO 风格）
    # Section 4.2: Horizon-level RL
    # 目标函数 Eq. 5: J = E[Σ γ^t r_sel - α × KL(â_sel || π_sel)]
    # â_sel 是 VQ encoder 对当前 horizon 示范轨迹分配的 ground-truth archetype label
    # PPO policy loss = -min(ratio*A, clip(ratio, 1±eps)*A)
    # Value loss = (R - V(s))²
    # imitation loss = KL(â_sel || π_sel) 的稳定实现
    # ----------------------------------------------------------------
    alpha = config.selection_alpha  # KL / imitation 惩罚系数
    total_steps = int(config.phase2_total_steps)
    val_interval = max(
        train_env.num_horizons,
        train_env.num_horizons * int(ppo_hparams["val_interval_multiplier"]),
    )
    log_interval = int(ppo_hparams["log_interval"])

    save_dir = config.get_stage_result_dir(pair, "phase2_archetype_selection")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{pair}_selection_agent.pt")

    best_val_return, reward_history, step_count = run_training_loop(
        agent=agent,
        encoder=encoder,
        codebook=codebook,
        decoder=decoder,
        eval_agent=eval_agent,
        eval_encoder=eval_encoder,
        eval_codebook=eval_codebook,
        eval_decoder=eval_decoder,
        eval_device=eval_device,
        train_env=train_env,
        val_env=val_env,
        demo_states=demo_states,
        demo_actions=demo_actions,
        demo_rewards=demo_rewards,
        optimizer=optimizer,
        critic_optimizer=critic_optimizer,
        alpha=alpha,
        total_steps=total_steps,
        val_interval=val_interval,
        log_interval=log_interval,
        save_path=save_path,
        config=config,
        ppo_hparams=ppo_hparams,
        device=device,
    )

    final_save_path = os.path.join(save_dir, f"{pair}_selection_agent_final.pt")
    save_checkpoint(
        save_path=final_save_path,
        agent=agent,
        optimizer=optimizer,
        critic_optimizer=critic_optimizer,
        reward_history=reward_history,
        best_val_return=best_val_return,
        step_count=step_count,
        config=config,
        ppo_hparams=ppo_hparams,
    )
    logger.info("最终模型已保存到 %s", final_save_path)

    # ----------------------------------------------------------------
    # Step 5: 训练结束后立即做一次验证集评估（best/final 对照）
    # 目标: 在进入 Phase III 前确认 Phase II 模型是否健康
    # ----------------------------------------------------------------
    eval_max_horizons = ppo_hparams["eval_max_horizons"]

    eval_agent.load_state_dict(agent.state_dict(), strict=True)
    final_val_metrics = evaluate_on_validation(
        agent=eval_agent,
        codebook=eval_codebook,
        decoder=eval_decoder,
        val_env=val_env,
        device=eval_device,
        max_horizons=eval_max_horizons,
    )
    final_val_return = float(final_val_metrics["avg_return"])

    best_val_metrics: dict[str, Any]
    if os.path.exists(save_path):
        best_ckpt = torch.load(save_path, map_location=eval_device, weights_only=False)
        best_agent = SelectionAgent(
            state_dim=config.state_dim,
            num_archetypes=config.num_archetypes,
            hidden_dim=config.phase2_hidden_dim,
            bottleneck_dim=config.phase2_bottleneck_dim,
        ).to(eval_device)
        best_agent.load_state_dict(best_ckpt["agent"])
        best_agent.eval()
        best_val_metrics = evaluate_on_validation(
            agent=best_agent,
            codebook=eval_codebook,
            decoder=eval_decoder,
            val_env=val_env,
            device=eval_device,
            max_horizons=eval_max_horizons,
        )
    else:
        logger.warning("未找到最优 checkpoint: %s，将使用最终模型验证结果替代", save_path)
        best_val_metrics = dict(final_val_metrics)

    best_status, best_status_msg = phase2_health_status(best_val_metrics)
    final_status, final_status_msg = phase2_health_status(final_val_metrics)

    logger.info(
        "Phase II 结束验证（BEST）: avg_return=%.4f, gross=%.4f, cost=%.4f, turnover=%.4f, flips=%.4f, selected=%s",
        float(best_val_metrics.get("avg_return", 0.0)),
        float(best_val_metrics.get("avg_gross_pnl", 0.0)),
        float(best_val_metrics.get("avg_execution_cost", 0.0)),
        float(best_val_metrics.get("avg_turnover", 0.0)),
        float(best_val_metrics.get("avg_direct_flips", 0.0)),
        best_val_metrics.get("selected_histogram", "[]"),
    )
    logger.info("Phase II 结束验证（BEST）健康度: %s — %s", best_status, best_status_msg)

    logger.info(
        "Phase II 结束验证（FINAL）: avg_return=%.4f, gross=%.4f, cost=%.4f, turnover=%.4f, flips=%.4f, selected=%s",
        float(final_val_metrics.get("avg_return", 0.0)),
        float(final_val_metrics.get("avg_gross_pnl", 0.0)),
        float(final_val_metrics.get("avg_execution_cost", 0.0)),
        float(final_val_metrics.get("avg_turnover", 0.0)),
        float(final_val_metrics.get("avg_direct_flips", 0.0)),
        final_val_metrics.get("selected_histogram", "[]"),
    )
    logger.info("Phase II 结束验证（FINAL）健康度: %s — %s", final_status, final_status_msg)

    if best_status != "healthy":
        logger.warning(
            "Phase II 最优模型健康度=%s。建议先修正第二阶段，再投入第三阶段训练。",
            best_status,
        )

    phase2_report = {
        "pair": pair,
        "step_count": int(step_count),
        "eval_max_horizons": eval_max_horizons,
        "best_checkpoint_path": save_path,
        "final_checkpoint_path": final_save_path,
        "best_checkpoint_validation": {
            **best_val_metrics,
            "health_status": best_status,
            "health_message": best_status_msg,
        },
        "final_checkpoint_validation": {
            **final_val_metrics,
            "health_status": final_status,
            "health_message": final_status_msg,
        },
    }
    phase2_report_path = os.path.join(save_dir, f"{pair}_phase2_validation_report.json")
    with open(phase2_report_path, "w", encoding="utf-8") as f:
        json.dump(phase2_report, f, ensure_ascii=False, indent=2)
    logger.info("Phase II 结束验证报告已保存到 %s", phase2_report_path)

    logger.info("=" * 50)
    logger.info("Phase II 训练完成: pair=%s", pair)
    logger.info("总训练步数: %d", step_count)
    logger.info(
        "最终平均奖励 (最近 1000 步): %.4f",
        np.mean(reward_history[-1000:]) if reward_history else float("nan"),
    )
    logger.info("训练期间最优验证集 return: %.4f", best_val_return)
    logger.info("结束时 FINAL 模型验证集 return: %.4f", final_val_return)
    logger.info("最优模型路径: %s", save_path)
    logger.info("最终模型路径: %s", final_save_path)
    logger.info("=" * 50)

    if bool(getattr(config, "phase2_stop_on_unhealthy", False)) and best_status != "healthy":
        logger.error(
            "已启用 --phase2-stop-on-unhealthy，且 Phase II 最优模型健康度=%s，训练流程在 Phase II 后终止。",
            best_status,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
