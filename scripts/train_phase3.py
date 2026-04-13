#!/usr/bin/env python
"""Phase III 训练脚本 — 原型精炼 (PPO)

# 需求: 7.3, 6.4, 6.7, 6.8, 7.4, 7.5, 7.6, 7.7
#
# 整体思路:
#   Phase I 学到了"原型动作序列"（码本），Phase II 学会了"选哪个原型"。
#   Phase III 在此基础上训练一个 RefinementAgent，它在每个交易步骤观察
#   当前市场状态和上下文，决定是否对 base action 做一次微调（加仓/减仓/不变）。
#   每个 horizon 最多允许一次调整（PolicyAdapter 强制约束）。
#
# 训练流程（每轮 num_envs 个 horizon）:
#   1. 随机采样 horizon，用冻结的 SelectionAgent + Decoder 生成 base actions
#   2. 先跑一遍 base rollout，收集各步 state 和 R_base（基线收益）
#   3. 用收集到的 states 批量前向 RefinementAgent，一次拿到所有步的决策
#      （避免逐步单独前向，提升 GPU 利用率）
#   4. 再跑一遍 refinement rollout，执行精炼后的动作，记录实际收益
#   5. 计算 hindsight-optimal 标签和 regret-aware reward
#   6. 把 num_envs 个 horizon 的数据合并成一个大 batch，做一次 PPO 更新
#      （合并 batch 是提升 GPU 利用率的关键，避免每次只更新 h=72 步的小 batch）
#
# 用法:
#   python scripts/train_phase3.py --pair BTC
#   python scripts/train_phase3.py --pair ETH --beta1 0.3 --phase3-total-steps 500000
#   python scripts/train_phase3.py --pair BTC --phase3-num-envs 16  # GPU 更强时加大并行数
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
from src.utils.gpu_guard import log_and_guard_gpu_memory, reset_gpu_peak_memory_stats
from src.utils.logger import get_logger
from src.utils.normalizer import StateNormalizer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------

@dataclass
class RolloutResult:
    """单个 horizon rollout 的全部输出。

    rollout 结束后，这些数据会被攒到 num_envs 个 horizon 的大 batch 里，
    再统一送入 ppo_update。字段说明：
      actual_return   : 精炼后动作序列的实际 horizon 总收益 R
      old_log_probs   : rollout 时采样动作的 log π_old(a|s)，PPO ratio 分母
      old_values      : rollout 时 critic 估计的 V(s)，用于计算 advantage
      a_refs          : 映射后的调整信号 ∈ {-1, 0, 1}，用于 regret reward 计算
      a_ref_indices   : 原始动作索引 ∈ {0, 1, 2}，用于 PPO log_prob 重算
      adjusted_step   : 实际生效的调整发生在哪一步（-1 表示未调整）
      cached_s_ref1/2 : 对应每步的市场观测和上下文 tensor，PPO epoch 重算时复用
    """
    actual_return: float
    old_log_probs: List[torch.Tensor]   # detached，shape 列表，每个 ()
    old_values: List[torch.Tensor]      # detached，shape 列表，每个 ()
    a_refs: List[int]                   # mapped a_ref ∈ {-1, 0, 1}
    a_ref_indices: List[int]            # raw action index ∈ {0, 1, 2}
    adjusted_step: int                  # -1 if no adjustment
    cached_s_ref1: List[torch.Tensor]   # 每个 (1, market_dim)
    cached_s_ref2: List[torch.Tensor]   # 每个 (1, context_dim)


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
        hidden_dim=config.phase2_hidden_dim,
        bottleneck_dim=config.phase2_bottleneck_dim,
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
    """用冻结的 Decoder 把原型向量 z_q 解码为 base action 序列 (h,)。

    Decoder 是 Phase I 训练好的 LSTM，输入整个 horizon 的 states 和原型向量，
    输出满足"单次交易约束"的动作序列。这里 batch=1（单个 horizon）。
    """
    states_t = torch.tensor(
        horizon_states, dtype=torch.float32, device=device,
    ).unsqueeze(0)  # (1, h, state_dim)
    with torch.no_grad():
        actions = decoder.decode_with_single_trade_constraint(
            states_t, z_q,
        ).squeeze(0)  # (h,)
    return actions.cpu().numpy()


def compute_base_return(
    env: TradingEnv, horizon_idx: int, base_actions: np.ndarray,
) -> float:
    """执行 base actions 并返回 horizon 总收益 R_base。

    注意: 此函数已不在主训练循环中调用。
    R_base 现在由 run_horizon_with_refinement 的第一遍 base rollout 顺带计算，
    避免对同一个 horizon 重复跑两遍 env.step。
    保留此函数供调试/验证使用。
    """
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
    """用冻结的 SelectionAgent 选择原型，返回 (z_q, e_a_sel_numpy)。

    SelectionAgent 看 horizon 第一步的市场状态，输出各原型的概率分布，
    取 argmax 选出最匹配的原型索引 k，再从码本里取出对应的嵌入向量 e_a_sel。
    z_q 送给 Decoder 生成 base actions；e_a_sel 作为 context 的一部分送给 RefinementAgent。
    """
    s0 = torch.tensor(state_0, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        probs, _ = selection_agent(s0)
        k = torch.argmax(probs, dim=-1).item()
    e_a_sel_t = codebook.embeddings.weight[k]  # 选中原型的嵌入向量
    z_q = e_a_sel_t.unsqueeze(0)               # (1, latent_dim)，送给 Decoder
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
) -> Tuple[RolloutResult, float]:
    """在一个 horizon 内执行 step 级别的精炼 rollout，同时计算 R_base。

    # 为什么分两遍跑？
    #
    # 原始实现每步单独调用 refinement_agent(s_ref1, s_ref2)（batch=1），
    # 72 步 = 72 次 GPU 前向，每次 kernel launch 开销远大于实际计算量，GPU 几乎空转。
    #
    # 改进方案：
    #   第一遍（base rollout）: 用 base_actions 跑环境，收集所有步的 state 和逐步奖励。
    #                           同时累积 R_base，省去原来单独调用 compute_base_return 的
    #                           第二次 env.step 遍历。
    #   批量前向:               把 h_eff 步的 states 和 context 堆叠成 (h_eff, dim) 的矩阵，
    #                           一次 GPU 前向拿到所有步的 action_probs 和 values。
    #   第二遍（refinement rollout）: env.step 有状态依赖必须串行，但 agent 决策已经算好，
    #                           直接按索引取结果，不再有 GPU 前向开销。

    Returns:
        (RolloutResult, R_base): rollout 数据 + base actions 的基线收益
    """
    h = len(base_actions)
    t_start = horizon_idx * env.horizon
    # notional = m × p_0，用于归一化累积奖励，使 context 向量的数值范围稳定
    notional = float(env.m) * float(env.prices[t_start])
    if notional <= 0.0:
        notional = 1.0

    # -----------------------------------------------------------------------
    # 第一遍: base rollout
    # 目的: 收集各步的市场 state（供批量前向用）+ 计算 R_base（基线收益）
    # 注意: state 在 env.step 之前记录，即 s_t 对应执行 a_t 之前的观测
    # -----------------------------------------------------------------------
    base_states: List[np.ndarray] = []
    base_step_rewards: List[float] = []
    state = env.reset(horizon_idx)
    for a in base_actions:
        base_states.append(state)                       # 记录执行前的 state
        state, reward, done, _ = env.step(int(a))
        base_step_rewards.append(reward)
        if done:
            break

    h_eff = len(base_states)          # 实际执行的步数（可能因 done 提前终止）
    R_base = sum(base_step_rewards)   # base actions 的 horizon 总收益

    # -----------------------------------------------------------------------
    # 构造 context 向量 s_ref2（每步一个）
    # s_ref2 = [e_a_sel (latent_dim) | a_base (1) | normalized_R_arche (1) | τ_remain (1)]
    # - e_a_sel: 选中原型的嵌入，告诉 agent "当前在执行哪个原型策略"
    # - a_base: 当前步的 base action，agent 以此为参考决定是否调整
    # - normalized_R_arche: 到当前步为止的累积收益（用 notional 归一化），反映执行进度
    # - τ_remain: 剩余时间比例，让 agent 感知"还有多少步可以操作"
    # 注意: normalized_R_arche 用的是 base rollout 的累积奖励，而非 refinement rollout 的，
    #       因为 context 在批量前向时已经固定，这是一个合理的近似。
    # -----------------------------------------------------------------------
    cum_r = 0.0
    context_rows: List[np.ndarray] = []
    for step_idx in range(h_eff):
        a_base = int(base_actions[step_idx])
        ctx = np.empty(len(e_a_sel) + 3, dtype=np.float32)
        ctx[:len(e_a_sel)] = e_a_sel
        ctx[-3] = float(a_base)
        ctx[-2] = cum_r / notional          # normalized_R_arche（执行前的累积，不含本步）
        ctx[-1] = float(h - step_idx) / h  # τ_remain
        context_rows.append(ctx)
        cum_r += base_step_rewards[step_idx]  # 本步奖励在构造完 context 后才累加

    # -----------------------------------------------------------------------
    # 批量前向: 一次 GPU 调用拿到所有步的决策
    # batch_s_ref1: (h_eff, state_dim)  — 市场观测
    # batch_s_ref2: (h_eff, context_dim) — 上下文
    # torch.no_grad(): rollout 阶段不需要梯度，节省显存和计算
    # -----------------------------------------------------------------------
    batch_s_ref1 = torch.tensor(
        np.stack(base_states, axis=0), dtype=torch.float32, device=device,
    )  # (h_eff, state_dim)
    batch_s_ref2 = torch.tensor(
        np.stack(context_rows, axis=0), dtype=torch.float32, device=device,
    )  # (h_eff, context_dim)

    with torch.no_grad():
        all_probs, all_values = refinement_agent(batch_s_ref1, batch_s_ref2)
    dist_batch = torch.distributions.Categorical(all_probs)
    a_ref_indices_t = dist_batch.sample()            # (h_eff,) 采样动作索引
    log_probs_t = dist_batch.log_prob(a_ref_indices_t)  # (h_eff,) 对应 log 概率

    # -----------------------------------------------------------------------
    # 第二遍: refinement rollout
    # env.step 依赖上一步的持仓状态，必须串行执行，无法并行化。
    # 但 agent 的决策已经在批量前向里算好，这里只是按索引取结果。
    # -----------------------------------------------------------------------
    state = env.reset(horizon_idx)
    has_adjusted = False   # 每个 horizon 最多一次调整（PolicyAdapter 约束）
    adjusted_step = -1     # 记录实际生效的调整发生在哪步，-1 表示未调整
    actual_return = 0.0

    old_log_probs: List[torch.Tensor] = []
    old_values: List[torch.Tensor] = []
    a_refs: List[int] = []
    a_ref_indices: List[int] = []

    a_base_prev = int(base_actions[0])

    for step_idx in range(h_eff):
        a_base = int(base_actions[step_idx])
        a_ref_idx = a_ref_indices_t[step_idx]
        a_ref = a_ref_idx.item() - 1  # 动作索引映射: 0→-1(减仓), 1→0(不变), 2→+1(加仓)

        prev_has_adjusted = has_adjusted
        # PolicyAdapter 实现 Eq.6 约束: base action 变化时不允许精炼，且每 horizon 最多一次调整
        a_final, has_adjusted = policy_adapter.compute_final_action(
            a_base, a_base_prev, a_ref, has_adjusted,
        )

        state, reward, done, _ = env.step(a_final)
        actual_return += reward

        # 保存 rollout 数据供 PPO 更新使用
        old_log_probs.append(log_probs_t[step_idx].detach())
        old_values.append(all_values[step_idx].squeeze().detach())
        a_refs.append(a_ref)
        a_ref_indices.append(a_ref_idx.item())

        a_base_prev = a_base

        if done:
            break

        # 首次调整后，RL episode 终止（不再收集 on-policy 数据），
        # 但继续用 base actions 跑完剩余步以计算完整 horizon 收益 R
        if has_adjusted and not prev_has_adjusted:
            adjusted_step = step_idx
            for remaining_idx in range(step_idx + 1, h_eff):
                _, r_rem, d_rem, _ = env.step(int(base_actions[remaining_idx]))
                actual_return += r_rem
                if d_rem:
                    break
            break

    rollout = RolloutResult(
        actual_return=actual_return,
        old_log_probs=old_log_probs,
        old_values=old_values,
        a_refs=a_refs,
        a_ref_indices=a_ref_indices,
        adjusted_step=adjusted_step,
        # 用切片保留 (1, dim) 形状，方便后续 torch.cat(dim=0) 拼接成大 batch
        cached_s_ref1=[batch_s_ref1[i:i+1] for i in range(len(a_refs))],
        cached_s_ref2=[batch_s_ref2[i:i+1] for i in range(len(a_refs))],
    )
    return rollout, R_base


# ---------------------------------------------------------------------------
# Reward / Label 计算
# ---------------------------------------------------------------------------

def build_hindsight_labels(
    top5: list, h_actual: int,
) -> np.ndarray:
    """从 top-5 hindsight-optimal adaptations 构建监督标签 â_ref（Eq. 9）。

    这是一种"事后诸葛亮"监督：在 horizon 结束后，枚举所有可能的调整点和调整方向，
    找出收益最高的 5 个，作为 CE loss 的监督目标，引导 agent 学会"在正确的时机调整"。

    返回 action index 数组 (h_actual,)：
      - 默认填 1（对应 a_ref=0，不调整）
      - 在 τ_opt 位置填入 a_opt+1（映射 {-1→0, +1→2}）
      - 若同一步有多个 top5 命中，只取第一个（收益最高的）
    """
    labels = np.ones(h_actual, dtype=np.int64)  # 默认: 不调整
    for tau_opt, a_opt, _ in top5:
        if tau_opt < h_actual and labels[tau_opt] == 1:  # 未被更高收益的覆盖
            labels[tau_opt] = a_opt + 1  # {-1→0, +1→2}
    return labels


def compute_step_rewards(
    a_refs: List[int], adjusted_step: int,
    R_actual: float, R_base: float, R_1_opt: float, beta1: float,
) -> List[float]:
    """计算每步的 regret-aware reward（Eq. 8）。

    奖励只在实际生效的调整步（adjusted_step）赋非零值，其余步为 0。
    这是一种稀疏奖励设计：agent 只在"做了调整"的那一步获得反馈，
    反馈值 = (R - R_base) + β₁ × (R - R₁_opt)，
    即"比基线好多少"减去"与最优解的差距"的加权组合。
    若未发生调整（adjusted_step=-1），所有步奖励均为 0。
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
    batch_s_ref1: torch.Tensor,   # (N, market_dim)，N = num_envs × h_actual
    batch_s_ref2: torch.Tensor,   # (N, context_dim)
    actions_t: torch.Tensor,      # (N,) rollout 时采样的动作索引
    old_log_probs_t: torch.Tensor,# (N,) rollout 时的 log π_old(a|s)
    returns_t: torch.Tensor,      # (N,) 折扣回报 G_t
    advantages: torch.Tensor,     # (N,) 归一化后的 advantage
    optimal_actions_t: torch.Tensor, # (N,) hindsight 监督标签
    *,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    beta2: float,
    max_grad_norm: float,
    ppo_epochs: int,
) -> float:
    """对合并后的大 batch 执行多轮 PPO 更新。

    # 为什么要多轮（ppo_epochs）？
    # PPO 的核心思想是用 clipped ratio 限制策略更新幅度，
    # 允许对同一批数据重复训练多次（通常 4 轮），提高样本利用率。
    # ratio = π_new(a|s) / π_old(a|s)，clip 到 [1-ε, 1+ε] 防止更新过大。
    #
    # 总损失 = policy_loss + vf_coef × value_loss + beta2 × ce_loss - ent_coef × entropy
    #   policy_loss: PPO clipped surrogate，最大化 advantage 加权的 log 概率
    #   value_loss:  critic 的 MSE 损失，让 V(s) 逼近实际回报 G_t
    #   ce_loss:     hindsight 监督（Eq. 9），用事后最优标签做 CE，加速收敛
    #   entropy:     熵正则化，鼓励探索，防止策略过早收敛到确定性动作

    Returns:
        最后一轮的 total loss 值（用于日志监控）。
    """
    last_loss = 0.0

    for _ in range(ppo_epochs):
        # 用当前策略重新评估 rollout 时的动作，计算新的 log 概率和价值
        action_probs, values = refinement_agent(batch_s_ref1, batch_s_ref2)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        # PPO clipped surrogate objective（Eq. 3 in PPO paper）
        # ratio > 1+ε 时 clip，防止单步更新过大导致策略崩溃
        ratio = torch.exp(new_log_probs - old_log_probs_t)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Critic loss: 让价值估计逼近实际折扣回报
        value_loss = F.mse_loss(values.squeeze(-1), returns_t)

        # Hindsight CE loss（Eq. 9）: 用事后最优标签做额外监督
        # 注意: action_probs 已经是 softmax 后的概率，cross_entropy 期望 logits，
        # 所以这里取 log 转回 log-space（等价于 NLLLoss）
        ce_loss = F.cross_entropy(torch.log(action_probs + 1e-8), optimal_actions_t)

        loss = (
            policy_loss
            + vf_coef * value_loss
            + beta2 * ce_loss
            - ent_coef * entropy
        )

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止梯度爆炸，对小模型（hidden=128）尤其重要
        torch.nn.utils.clip_grad_norm_(refinement_agent.parameters(), max_grad_norm)
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
    reset_gpu_peak_memory_stats(device)
    log_and_guard_gpu_memory(logger, stage="Phase III startup", device=device, force_log=False)

    # --- 加载冻结模型 ---
    codebook, decoder, normalizer = load_phase1_model(config, pair, device)
    selection_agent = load_phase2_model(config, pair, device)
    log_and_guard_gpu_memory(logger, stage="Phase III after frozen model load", device=device, force_log=False)

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
        commission_rate=config.train_commission_rate,
    )
    logger.info("TradingEnv: train_horizons=%d", train_env.num_horizons)
    log_and_guard_gpu_memory(logger, stage="Phase III after env init", device=device, force_log=False)
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
    log_and_guard_gpu_memory(logger, stage="Phase III after agent init", device=device, force_log=False)

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
    num_envs = config.phase3_num_envs  # 每轮并行收集的 horizon 数
    log_interval = 100

    logger.info("每轮并行 horizon 数: %d (有效 batch_size ≈ %d)", num_envs, num_envs * config.horizon)

    # --- 输出路径 ---
    save_dir = os.path.join(
        config.get_stage_result_dir(pair, "phase3_archetype_refinement"),
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"{pair}_refinement_agent_beta{beta1}.pt",
    )

    # --- 训练循环 ---
    # policy_adapter 是纯函数设计（无内部状态），复用同一个实例即可
    policy_adapter = PolicyAdapter()
    reward_history: List[float] = []
    step_count = 0
    horizon_count = 0
    last_loss = 0.0

    logger.info("开始训练: %d 步", total_steps)
    pbar = tqdm(total=total_steps, desc="Phase III", unit="step", dynamic_ncols=True)

    pbar = tqdm(total=total_steps, desc="Phase III", unit="step", dynamic_ncols=True)

    while step_count < total_steps:
        # ===================================================================
        # 数据收集阶段: 串行跑 num_envs 个 horizon，攒成一个大 batch
        #
        # 为什么不用真正的并行（multiprocessing）？
        #   TradingEnv 是有状态的 Python 对象，跨进程共享成本高。
        #   当前瓶颈是 GPU 利用率不足（batch 太小），串行收集 + 合并 batch
        #   已经能显著提升 GPU 利用率，实现简单且无同步开销。
        # ===================================================================
        batch_returns_t: List[torch.Tensor] = []
        batch_old_log_probs: List[torch.Tensor] = []
        batch_old_values: List[torch.Tensor] = []
        batch_actions: List[torch.Tensor] = []
        batch_s_ref1_all: List[torch.Tensor] = []
        batch_s_ref2_all: List[torch.Tensor] = []
        batch_optimal_actions: List[torch.Tensor] = []
        steps_this_round = 0

        for _ in range(num_envs):
            # 1) 随机采样一个 horizon（有放回，保证每轮都能凑满 num_envs 个）
            h_idx = np.random.randint(0, train_env.num_horizons)
            h = train_env.horizon
            start = h_idx * h
            end = min(start + h, len(train_env.states))
            horizon_states = train_env.states[start:end]
            horizon_prices = train_env.prices[start:end]

            # 2) 冻结模型推断: 选原型 → 生成 base actions
            z_q, e_a_sel = select_archetype(
                selection_agent, codebook, train_env.states[start], device,
            )
            base_actions = generate_base_actions(decoder, z_q, horizon_states, device)

            # 3) Rollout: 两遍跑环境 + 一次批量前向
            #    返回 rollout 数据和 R_base（base actions 的基线收益）
            rollout, R_base = run_horizon_with_refinement(
                env=train_env, horizon_idx=h_idx, base_actions=base_actions,
                refinement_agent=refinement_agent, policy_adapter=policy_adapter,
                e_a_sel=e_a_sel, device=device,
            )
            h_actual = len(rollout.a_refs)
            if h_actual == 0:
                continue  # 极端情况：horizon 第一步就 done，跳过

            # 4) Hindsight-optimal labels（Eq. 7 & 9）
            #    事后枚举所有可能的调整点，找出收益最高的 5 个作为监督标签
            states_list = train_env.states_dataframe[start:end].rows(named=True)
            top5 = compute_top5_hindsight_optimal(
                prices=horizon_prices, base_actions=base_actions,
                step_idx=0, env=train_env, states=states_list,
            )
            R_1_opt = top5[0][2] if top5 else R_base  # top-1 最优收益，用于 regret 计算
            optimal_actions = build_hindsight_labels(top5, h_actual)

            # 5) 计算稀疏奖励 → 折扣回报
            #    只有 adjusted_step 处有非零奖励，其余步为 0
            step_rewards = compute_step_rewards(
                rollout.a_refs, rollout.adjusted_step,
                rollout.actual_return, R_base, R_1_opt, beta1,
            )
            returns = compute_discounted_returns(step_rewards, gamma)

            # 把本 horizon 的数据追加到本轮 batch 列表
            batch_returns_t.append(torch.tensor(returns, dtype=torch.float32, device=device))
            batch_old_log_probs.append(torch.stack(rollout.old_log_probs))
            batch_old_values.append(torch.stack(rollout.old_values))
            batch_actions.append(torch.tensor(rollout.a_ref_indices, dtype=torch.long, device=device))
            batch_s_ref1_all.append(torch.cat(rollout.cached_s_ref1, dim=0))
            batch_s_ref2_all.append(torch.cat(rollout.cached_s_ref2, dim=0))
            batch_optimal_actions.append(torch.tensor(optimal_actions, dtype=torch.long, device=device))

            steps_this_round += h_actual
            reward_history.append(rollout.actual_return)

        if not batch_returns_t:
            continue

        force_gpu_log = (
            horizon_count == 0
            or ((horizon_count + len(batch_returns_t)) % log_interval == 0)
            or (step_count + steps_this_round >= total_steps)
        )

        # ===================================================================
        # 更新阶段: 把 num_envs 个 horizon 的数据合并成一个大 batch，做一次 PPO
        #
        # 合并后 batch size ≈ num_envs × h_actual（通常 8×72=576）
        # 相比原来每个 horizon 单独更新（batch=72），GPU 利用率提升约 num_envs 倍
        # ===================================================================
        returns_t      = torch.cat(batch_returns_t)       # (N,)
        old_log_probs_t = torch.cat(batch_old_log_probs)  # (N,)
        old_values_t   = torch.cat(batch_old_values)      # (N,)
        actions_t      = torch.cat(batch_actions)         # (N,)
        s_ref1_t       = torch.cat(batch_s_ref1_all)      # (N, market_dim)
        s_ref2_t       = torch.cat(batch_s_ref2_all)      # (N, context_dim)
        optimal_actions_t = torch.cat(batch_optimal_actions)  # (N,)
        log_and_guard_gpu_memory(
            logger,
            stage=(
                "Phase III after batch assembly"
                f"(step={step_count}, steps_this_round={steps_this_round}, horizons={len(batch_returns_t)})"
            ),
            device=device,
            force_log=force_gpu_log,
        )

        # Advantage 归一化: 减均值除标准差，稳定训练
        advantages = returns_t - old_values_t
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        last_loss = ppo_update(
            refinement_agent, optimizer,
            s_ref1_t, s_ref2_t,
            actions_t, old_log_probs_t, returns_t, advantages,
            optimal_actions_t,
            clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef,
            beta2=beta2, max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs,
        )
        log_and_guard_gpu_memory(
            logger,
            stage=(
                "Phase III after ppo_update"
                f"(step={step_count}, steps_this_round={steps_this_round}, horizons={len(batch_returns_t)})"
            ),
            device=device,
            force_log=force_gpu_log,
        )

        if np.isnan(last_loss):
            logger.error("训练 loss 发散 (NaN)，在 step %d 终止", step_count)
            break

        # 计数 & 日志
        step_count += steps_this_round
        horizon_count += len(batch_returns_t)
        pbar.update(steps_this_round)

        if horizon_count % log_interval == 0:
            avg = np.mean(reward_history[-log_interval:])
            tqdm.write(
                "Step %7d/%d (horizon %d) — avg=%.4f, loss=%.4f"
                % (step_count, total_steps, horizon_count, avg, last_loss)
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
