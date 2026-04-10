#!/usr/bin/env python
"""Phase III 训练脚本 — 原型精炼 (PPO)

# 需求: 7.3, 6.4, 6.7, 6.8, 7.4, 7.5, 7.6, 7.7
#
# 流程:
# 1. 加载 Phase I 模型（码本 + 冻结 Decoder）和 Phase II 模型（冻结 SelectionAgent）
# 2. 加载特征数据，初始化 TradingEnv（训练集）
# 3. 初始化 RefinementAgent（Actor-Critic + AdaLN）、PolicyAdapter
# 4. 训练 1M 步（step 级别 RL，PPO 风格）
#    - 每个 horizon: selection agent 选择原型 → decoder 生成 base actions
#    - 每个 step: refinement agent 观测 state + context → 输出 a_ref
#    - Horizon 结束: regret-aware reward + PPO clipped objective 更新
# 5. 保存模型到 result/phase3_archetype_refinement/
#
# 用法:
#   python scripts/train_phase3.py --pair BTC
#   python scripts/train_phase3.py --pair ETH --beta1 0.3 --phase3-total-steps 500000
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import parse_args
from src.data.feature_pipeline import FeaturePipeline
from src.env.trading_env import TradingEnv
from src.phase1.codebook import VQCodebook
from src.phase1.vq_decoder import VQDecoder
from src.phase2.selection_agent import SelectionAgent
from src.phase3.refinement_agent import RefinementAgent
from src.phase3.policy_adapter import PolicyAdapter
from src.phase3.regret_reward import compute_regret_reward, compute_top5_hindsight_optimal
from src.utils.logger import get_logger
from src.utils.normalizer import StateNormalizer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------

@dataclass
class RolloutResult:
    """单个 horizon rollout 的全部输出。"""
    actual_return: float
    old_log_probs: List[torch.Tensor]   # detached
    old_values: List[torch.Tensor]      # detached
    a_refs: List[int]                   # mapped a_ref ∈ {-1, 0, 1}
    a_ref_indices: List[int]            # raw action index ∈ {0, 1, 2}
    adjusted_step: int                  # -1 if no adjustment
    cached_s_ref1: List[torch.Tensor]
    cached_s_ref2: List[torch.Tensor]


# ---------------------------------------------------------------------------
# Phase I / II 模型加载
# ---------------------------------------------------------------------------

def load_phase1_model(config, pair: str, device: torch.device):
    """加载并冻结 Phase I 模型（码本 + Decoder）+ 归一化统计量。"""
    model_path = os.path.join(
        config.get_stage_result_dir(pair, "phase1_archetype_discovery"),
        f"{pair}_vq_model.pt",
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Phase I 模型文件不存在: {model_path}\n"
            f"请先运行 Phase I 训练: python scripts/train_phase1.py --pair {pair}"
        )

    logger.info("加载 Phase I 模型: %s", model_path)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    codebook = VQCodebook(
        num_codes=config.num_archetypes, code_dim=config.latent_dim,
    ).to(device)
    codebook.load_state_dict(ckpt["codebook"])

    decoder = VQDecoder(
        state_dim=config.state_dim, code_dim=config.latent_dim,
        hidden_dim=config.lstm_hidden_dim, action_dim=config.action_dim,
    ).to(device)
    decoder.load_state_dict(ckpt["decoder"])

    for p in codebook.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False
    codebook.eval()
    decoder.eval()

    normalizer = StateNormalizer.from_checkpoint_dict(ckpt)
    if normalizer is not None:
        logger.info("Phase I 归一化统计量已加载")

    logger.info("Phase I 模型加载完成，Codebook 和 Decoder 已冻结")
    return codebook, decoder, normalizer


def load_phase2_model(config, pair: str, device: torch.device):
    """加载并冻结 Phase II 模型（SelectionAgent）。"""
    model_path = os.path.join(
        config.get_stage_result_dir(pair, "phase2_archetype_selection"),
        f"{pair}_selection_agent.pt",
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Phase II 模型文件不存在: {model_path}\n"
            f"请先运行 Phase II 训练: python scripts/train_phase2.py --pair {pair}"
        )

    logger.info("加载 Phase II 模型: %s", model_path)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    agent = SelectionAgent(
        state_dim=config.state_dim, num_archetypes=config.num_archetypes,
    ).to(device)
    agent.load_state_dict(ckpt["agent"])

    for p in agent.parameters():
        p.requires_grad = False
    agent.eval()

    logger.info("Phase II 模型加载完成，SelectionAgent 已冻结")
    return agent


# ---------------------------------------------------------------------------
# Horizon 级别辅助函数
# ---------------------------------------------------------------------------

def generate_base_actions(
    decoder: VQDecoder, z_q: torch.Tensor,
    horizon_states: np.ndarray, device: torch.device,
) -> np.ndarray:
    """使用冻结 Decoder 生成 base actions (h,)。"""
    states_t = torch.tensor(
        horizon_states, dtype=torch.float32, device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        actions = decoder.decode_with_single_trade_constraint(
            states_t, z_q,
        ).squeeze(0)
    return actions.cpu().numpy()


def compute_base_return(
    env: TradingEnv, horizon_idx: int, base_actions: np.ndarray,
) -> float:
    """执行 base actions 并返回 horizon 总收益 R_base。"""
    env.reset(horizon_idx)
    total = 0.0
    for a in base_actions:
        _, reward, done, _ = env.step(int(a))
        total += reward
        if done:
            break
    return total


def select_archetype(
    selection_agent: SelectionAgent, codebook: VQCodebook,
    state_0: np.ndarray, device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    """冻结 SelectionAgent 选择原型，返回 (z_q, e_a_sel_numpy)。"""
    s0 = torch.tensor(state_0, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        probs, _ = selection_agent(s0)
        k = torch.argmax(probs, dim=-1).item()
    e_a_sel_t = codebook.embeddings.weight[k]
    z_q = e_a_sel_t.unsqueeze(0)
    return z_q, e_a_sel_t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Rollout: 在一个 horizon 内执行 step 级别精炼
# ---------------------------------------------------------------------------

def run_horizon_with_refinement(
    env: TradingEnv,
    horizon_idx: int,
    base_actions: np.ndarray,
    refinement_agent: RefinementAgent,
    policy_adapter: PolicyAdapter,
    e_a_sel: np.ndarray,
    device: torch.device,
) -> RolloutResult:
    """在一个 horizon 内执行 step 级别的精炼 rollout。

    收集 on-policy 数据供后续 PPO 更新使用。
    RL episode 在 adapter 首次选择非零 a_ref 后终止，
    但剩余 base actions 仍会执行以计算完整 horizon return R。
    """
    state = env.reset(horizon_idx)
    h = len(base_actions)

    # R_arche 归一化分母: m × p_0
    t_start = horizon_idx * env.horizon
    notional = float(env.m) * float(env.prices[t_start])
    if notional <= 0.0:
        notional = 1.0

    has_adjusted = False
    adjusted_step = -1
    actual_return = 0.0
    cumulative_reward = 0.0

    old_log_probs: List[torch.Tensor] = []
    old_values: List[torch.Tensor] = []
    a_refs: List[int] = []
    a_ref_indices: List[int] = []
    cached_s_ref1: List[torch.Tensor] = []
    cached_s_ref2: List[torch.Tensor] = []

    a_base_prev = int(base_actions[0])

    for step_idx in range(h):
        a_base = int(base_actions[step_idx])

        # s_ref1 = market observation
        s_ref1 = torch.tensor(
            state, dtype=torch.float32, device=device,
        ).unsqueeze(0)

        # s_ref2 = [e_a_sel, a_base, normalized_R_arche, τ_remain]
        tau_remain = float(h - step_idx) / h
        normalized_reward = cumulative_reward / notional
        context = np.concatenate([
            e_a_sel,
            np.array([a_base], dtype=np.float32),
            np.array([normalized_reward], dtype=np.float32),
            np.array([tau_remain], dtype=np.float32),
        ])
        s_ref2 = torch.tensor(
            context, dtype=torch.float32, device=device,
        ).unsqueeze(0)

        cached_s_ref1.append(s_ref1)
        cached_s_ref2.append(s_ref2)

        # Refinement agent forward
        action_probs, value = refinement_agent(s_ref1, s_ref2)
        dist = torch.distributions.Categorical(action_probs)
        a_ref_idx = dist.sample()
        log_prob = dist.log_prob(a_ref_idx)

        a_ref = a_ref_idx.item() - 1  # 0→-1, 1→0, 2→1

        # Policy adapter
        prev_has_adjusted = has_adjusted
        a_final, has_adjusted = policy_adapter.compute_final_action(
            a_base, a_base_prev, a_ref, has_adjusted,
        )

        # Env step
        next_state, reward, done, _ = env.step(a_final)
        actual_return += reward
        cumulative_reward += reward

        old_log_probs.append(log_prob.detach())
        old_values.append(value.squeeze().detach())
        a_refs.append(a_ref)
        a_ref_indices.append(a_ref_idx.item())

        state = next_state
        a_base_prev = a_base

        if done:
            break

        # RL episode 终止于首次非零调整，但继续执行 base actions 计算 R
        if has_adjusted and not prev_has_adjusted:
            adjusted_step = step_idx
            for remaining_idx in range(step_idx + 1, h):
                _, r_rem, d_rem, _ = env.step(int(base_actions[remaining_idx]))
                actual_return += r_rem
                if d_rem:
                    break
            break

    return RolloutResult(
        actual_return=actual_return,
        old_log_probs=old_log_probs,
        old_values=old_values,
        a_refs=a_refs,
        a_ref_indices=a_ref_indices,
        adjusted_step=adjusted_step,
        cached_s_ref1=cached_s_ref1,
        cached_s_ref2=cached_s_ref2,
    )


# ---------------------------------------------------------------------------
# Reward / Label 计算
# ---------------------------------------------------------------------------

def build_hindsight_labels(
    top5: list, h_actual: int,
) -> np.ndarray:
    """从 top-5 hindsight-optimal adaptations 构建监督标签 â_ref。

    返回 action index 数组 (h_actual,)，默认 1 (不调整)，
    在 τ_opt 位置填入 a_opt+1 (映射 {-1,1} → {0,2})。
    """
    labels = np.ones(h_actual, dtype=np.int64)
    for tau_opt, a_opt, _ in top5:
        if tau_opt < h_actual and labels[tau_opt] == 1:
            labels[tau_opt] = a_opt + 1
    return labels


def compute_step_rewards(
    a_refs: List[int], adjusted_step: int,
    R_actual: float, R_base: float, R_1_opt: float, beta1: float,
) -> List[float]:
    """计算每步的 regret-aware reward (Eq. 8)。

    仅在实际生效的调整步赋非零值，其余步为 0。
    """
    rewards = []
    for idx in range(len(a_refs)):
        if idx == adjusted_step:
            rewards.append(compute_regret_reward(
                R=R_actual, R_base=R_base, R_1_opt=R_1_opt,
                a_ref=a_refs[idx], beta1=beta1,
            ))
        else:
            rewards.append(0.0)
    return rewards


def compute_discounted_returns(
    step_rewards: List[float], gamma: float,
) -> List[float]:
    """从后向前计算折扣回报 G_t = r_t + γ G_{t+1}。"""
    returns = []
    G = 0.0
    for r in reversed(step_rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


# ---------------------------------------------------------------------------
# PPO 更新
# ---------------------------------------------------------------------------

def ppo_update(
    refinement_agent: RefinementAgent,
    optimizer: torch.optim.Optimizer,
    batch_s_ref1: torch.Tensor,
    batch_s_ref2: torch.Tensor,
    actions_t: torch.Tensor,
    old_log_probs_t: torch.Tensor,
    returns_t: torch.Tensor,
    advantages: torch.Tensor,
    optimal_actions_t: torch.Tensor,
    *,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    beta2: float,
    max_grad_norm: float,
    ppo_epochs: int,
) -> float:
    """对单个 horizon 的 rollout 数据执行多轮 PPO 更新。

    Returns:
        最后一轮的 total loss 值（用于日志）。
    """
    last_loss = 0.0

    for _ in range(ppo_epochs):
        # 重新评估当前策略
        action_probs, values = refinement_agent(batch_s_ref1, batch_s_ref2)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        # PPO clipped surrogate
        ratio = torch.exp(new_log_probs - old_log_probs_t)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - clip_eps, 1.0 + clip_eps,
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values.squeeze(-1), returns_t)

        # CE supervision loss (Eq. 9)
        ce_loss = F.cross_entropy(
            torch.log(action_probs + 1e-8), optimal_actions_t,
        )

        # 总损失
        loss = (
            policy_loss
            + vf_coef * value_loss
            + beta2 * ce_loss
            - ent_coef * entropy
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            refinement_agent.parameters(), max_grad_norm,
        )
        optimizer.step()
        last_loss = loss.item()

    return last_loss


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, agent: RefinementAgent, config, **extra):
    """保存 Phase III 模型 checkpoint。"""
    torch.save(
        {
            "agent": agent.state_dict(),
            "config": {
                "state_dim": config.state_dim,
                "latent_dim": config.latent_dim,
                "num_archetypes": config.num_archetypes,
                "phase3_total_steps": config.phase3_total_steps,
                "refinement_hidden_dim": config.refinement_hidden_dim,
                "refinement_beta1": config.refinement_beta1,
                "refinement_beta2": config.refinement_beta2,
                "phase3_clip_eps": config.phase3_clip_eps,
                "phase3_ppo_epochs": config.phase3_ppo_epochs,
                "phase3_vf_coef": config.phase3_vf_coef,
                "phase3_ent_coef": config.phase3_ent_coef,
                "phase3_max_grad_norm": config.phase3_max_grad_norm,
                "learning_rate": config.learning_rate,
                "discount_factor": config.discount_factor,
            },
            **extra,
        },
        path,
    )
    logger.info("模型已保存到 %s", path)


# ---------------------------------------------------------------------------
# 主训练循环
# ---------------------------------------------------------------------------

def main() -> None:
    # --- 配置 ---
    config = parse_args()
    pair = config.pairs[0]
    beta1 = config.refinement_beta1
    beta2 = config.refinement_beta2

    logger.info("Phase III 训练开始: pair=%s", pair)
    logger.info(
        "超参数: total_steps=%d, lr=%.1e, beta1=%.2f, beta2=%.2f, "
        "num_archetypes=%d, gamma=%.2f",
        config.phase3_total_steps, config.learning_rate,
        beta1, beta2, config.num_archetypes, config.discount_factor,
    )
    logger.info(
        "PPO 超参数: clip_eps=%.2f, ppo_epochs=%d, vf_coef=%.3f, "
        "ent_coef=%.4f, max_grad_norm=%.2f",
        config.phase3_clip_eps, config.phase3_ppo_epochs,
        config.phase3_vf_coef, config.phase3_ent_coef,
        config.phase3_max_grad_norm,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)
    logger.info("结果目录批次: %s", config.train_batch_id)

    # --- 加载冻结模型 ---
    codebook, decoder, normalizer = load_phase1_model(config, pair, device)
    selection_agent = load_phase2_model(config, pair, device)

    # --- 数据 & 环境 ---
    logger.info("加载特征数据: data_dir=%s, pair=%s", config.data_dir, pair)
    pipeline = FeaturePipeline(
        config.data_dir, pair, cycle_features=config.cycle_features,
    )
    train_df, _, _ = pipeline.get_state_vector()
    train_prices_df, _, _ = pipeline.get_prices()

    train_env = TradingEnv(
        states=normalizer.normalize_states(train_df.to_numpy()) if normalizer else train_df.to_numpy(),
        prices=train_prices_df["close"].to_numpy(),
        pair=pair,
        horizon=config.horizon,
        states_dataframe=train_df,
        max_positions=config.max_positions,
        commission_rate=config.commission_rate,
    )
    logger.info("TradingEnv: train_horizons=%d", train_env.num_horizons)
    if train_env.num_horizons == 0:
        logger.error("训练集 horizon 数量为 0，无法训练")
        sys.exit(1)

    # --- RefinementAgent ---
    context_dim = config.latent_dim + 3
    refinement_agent = RefinementAgent(
        market_dim=config.state_dim,
        context_dim=context_dim,
        hidden_dim=config.refinement_hidden_dim,
    ).to(device)
    logger.info(
        "RefinementAgent: params=%d, market_dim=%d, context_dim=%d",
        sum(p.numel() for p in refinement_agent.parameters()),
        config.state_dim, context_dim,
    )

    optimizer = torch.optim.Adam(
        refinement_agent.parameters(), lr=config.learning_rate,
    )

    # --- PPO 超参数 ---
    gamma = config.discount_factor
    clip_eps = config.phase3_clip_eps
    ppo_epochs = config.phase3_ppo_epochs
    vf_coef = config.phase3_vf_coef
    ent_coef = config.phase3_ent_coef
    max_grad_norm = config.phase3_max_grad_norm
    total_steps = config.phase3_total_steps
    log_interval = 100

    # --- 输出路径 ---
    save_dir = os.path.join(
        config.get_stage_result_dir(pair, "phase3_archetype_refinement"),
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"{pair}_refinement_agent_beta{beta1}.pt",
    )

    # --- 训练循环 ---
    reward_history: List[float] = []
    step_count = 0
    horizon_count = 0
    last_loss = 0.0

    logger.info("开始训练: %d 步", total_steps)
    pbar = tqdm(total=total_steps, desc="Phase III", unit="step", dynamic_ncols=True)

    while step_count < total_steps:
        # 1) 采样 horizon
        h_idx = np.random.randint(0, train_env.num_horizons)
        h = train_env.horizon
        start = h_idx * h
        end = min(start + h, len(train_env.states))
        horizon_states = train_env.states[start:end]
        horizon_prices = train_env.prices[start:end]

        # 2) 选择原型 + 生成 base actions
        z_q, e_a_sel = select_archetype(
            selection_agent, codebook, train_env.states[start], device,
        )
        base_actions = generate_base_actions(decoder, z_q, horizon_states, device)
        R_base = compute_base_return(train_env, h_idx, base_actions)

        # 3) Rollout: step 级别精炼
        rollout = run_horizon_with_refinement(
            env=train_env, horizon_idx=h_idx, base_actions=base_actions,
            refinement_agent=refinement_agent, policy_adapter=PolicyAdapter(),
            e_a_sel=e_a_sel, device=device,
        )
        h_actual = len(rollout.a_refs)

        # 4) Hindsight-optimal labels (Eq. 7 & 9)
        states_list = train_env.states_dataframe[start:end].rows(named=True)
        top5 = compute_top5_hindsight_optimal(
            prices=horizon_prices, base_actions=base_actions,
            step_idx=0, env=train_env, states=states_list,
        )
        R_1_opt = top5[0][2] if top5 else R_base
        optimal_actions = build_hindsight_labels(top5, h_actual)

        # 5) Regret-aware rewards (Eq. 8)
        step_rewards = compute_step_rewards(
            rollout.a_refs, rollout.adjusted_step,
            rollout.actual_return, R_base, R_1_opt, beta1,
        )

        # 6) PPO 更新
        if rollout.old_log_probs:
            returns = compute_discounted_returns(step_rewards, gamma)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
            old_log_probs_t = torch.stack(rollout.old_log_probs)
            old_values_t = torch.stack(rollout.old_values)
            actions_t = torch.tensor(
                rollout.a_ref_indices, dtype=torch.long, device=device,
            )

            advantages = returns_t - old_values_t
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            batch_s_ref1 = torch.cat(rollout.cached_s_ref1, dim=0)
            batch_s_ref2 = torch.cat(rollout.cached_s_ref2, dim=0)
            optimal_actions_t = torch.tensor(
                optimal_actions, dtype=torch.long, device=device,
            )

            last_loss = ppo_update(
                refinement_agent, optimizer,
                batch_s_ref1, batch_s_ref2,
                actions_t, old_log_probs_t, returns_t, advantages,
                optimal_actions_t,
                clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef,
                beta2=beta2, max_grad_norm=max_grad_norm,
                ppo_epochs=ppo_epochs,
            )

            if np.isnan(last_loss):
                logger.error("训练 loss 发散 (NaN)，在 step %d 终止", step_count)
                break

        # 7) 计数 & 日志
        step_count += h_actual
        horizon_count += 1
        reward_history.append(rollout.actual_return)
        pbar.update(h_actual)

        if horizon_count % log_interval == 0:
            avg = np.mean(reward_history[-log_interval:])
            tqdm.write(
                "Step %7d/%d (horizon %d) — avg=%.4f, R=%.4f, "
                "R_base=%.4f, R_1_opt=%.4f, loss=%.4f"
                % (step_count, total_steps, horizon_count, avg,
                   rollout.actual_return, R_base, R_1_opt, last_loss)
            )

    pbar.close()

    # --- 保存 ---
    save_checkpoint(
        save_path, refinement_agent, config,
        training_rewards=reward_history,
        step=step_count, beta1=beta1,
    )

    logger.info("=" * 50)
    logger.info("Phase III 训练完成: pair=%s, beta1=%.2f", pair, beta1)
    logger.info("总训练步数: %d, 总 horizon 数: %d", step_count, horizon_count)
    logger.info(
        "最终平均奖励 (最近 1000 horizons): %.4f",
        np.mean(reward_history[-1000:]) if reward_history else float("nan"),
    )
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
