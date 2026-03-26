#!/usr/bin/env python
"""Phase III 训练脚本 — 原型精炼

# 需求: 7.3, 6.4, 6.7, 6.8, 7.4, 7.5, 7.6, 7.7
#
# 流程:
# 1. 加载 Phase I 模型（码本 + 冻结 Decoder）和 Phase II 模型（冻结 SelectionAgent）
# 2. 加载特征数据，初始化 TradingEnv（训练集）
# 3. 初始化 RefinementAgent（Actor-Critic + AdaLN）、PolicyAdapter
# 4. 训练 1M 步（step 级别 RL）
#    - 每个 horizon: selection agent 选择原型 → decoder 生成 base actions
#    - 每个 step: refinement agent 观测 state + context → 输出 a_ref → policy adapter 计算 final action → env step
#    - Horizon 结束: 计算 regret-aware reward，更新 refinement agent (Actor-Critic)
#    - r_ref = (R - R_base) + β_1 × (R - R_1_opt)  if a_ref != 0
#    - r_ref = 0                                      if a_ref == 0
# 5. 保存模型到 result/phase3_archetype_refinement/
# 6. 输出训练日志
#
# 用法:
#   python scripts/train_phase3.py --pair BTC
#   python scripts/train_phase3.py --pair ETH --beta1 0.3 --phase3-total-steps 500000
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

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

logger = get_logger(__name__)


def load_phase1_model(config, pair: str, device: torch.device):
    """加载 Phase I 模型（码本 + 冻结 Decoder）。

    # 需求 7.4: 前置阶段模型文件不存在时抛出明确错误
    # 需求 5.3: 冻结 Codebook 和 Decoder 参数

    Returns:
        codebook: 加载权重后的 VQCodebook（冻结）
        decoder: 加载权重后的 VQDecoder（冻结）
    """
    model_path = os.path.join(
        config.result_dir, "phase1_archetype_discovery", f"{pair}_vq_model.pt"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Phase I 模型文件不存在: {model_path}\n"
            f"请先运行 Phase I 训练: python scripts/train_phase1.py --pair {pair}"
        )

    logger.info("加载 Phase I 模型: %s", model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 初始化并加载 Codebook
    codebook = VQCodebook(
        num_codes=config.num_archetypes,
        code_dim=config.latent_dim,
    ).to(device)
    codebook.load_state_dict(checkpoint["codebook"])

    # 初始化并加载 Decoder
    decoder = VQDecoder(
        state_dim=config.state_dim,
        code_dim=config.latent_dim,
        hidden_dim=config.lstm_hidden_dim,
        action_dim=config.action_dim,
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder"])

    # 冻结 Codebook 和 Decoder — 不参与梯度更新
    for param in codebook.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    codebook.eval()
    decoder.eval()

    logger.info("Phase I 模型加载完成，Codebook 和 Decoder 已冻结")
    return codebook, decoder


def load_phase2_model(config, pair: str, device: torch.device):
    """加载 Phase II 模型（冻结 SelectionAgent）。

    # 需求 7.4: 前置阶段模型文件不存在时抛出明确错误

    Returns:
        selection_agent: 加载权重后的 SelectionAgent（冻结）
    """
    model_path = os.path.join(
        config.result_dir, "phase2_archetype_selection", f"{pair}_selection_agent.pt"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Phase II 模型文件不存在: {model_path}\n"
            f"请先运行 Phase II 训练: python scripts/train_phase2.py --pair {pair}"
        )

    logger.info("加载 Phase II 模型: %s", model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    selection_agent = SelectionAgent(
        state_dim=config.state_dim,
        num_archetypes=config.num_archetypes,
    ).to(device)
    selection_agent.load_state_dict(checkpoint["agent"])

    # 冻结 SelectionAgent — 不参与梯度更新
    for param in selection_agent.parameters():
        param.requires_grad = False

    selection_agent.eval()

    logger.info("Phase II 模型加载完成，SelectionAgent 已冻结")
    return selection_agent


def generate_base_actions(
    decoder: VQDecoder,
    z_q: torch.Tensor,
    horizon_states: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """使用冻结 Decoder 生成 horizon 内的 base actions。

    # Section 4.2: 冻结 Decoder 生成 micro actions

    Args:
        decoder: 冻结的 VQ Decoder
        z_q: 选定原型的量化嵌入 (1, code_dim)
        horizon_states: horizon 内的状态序列 (h, state_dim)
        device: 计算设备

    Returns:
        base_actions: 基础动作序列 (h,)，值域 {0, 1, 2}
    """
    states_t = torch.tensor(
        horizon_states, dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, h, state_dim)

    with torch.no_grad():
        action_logits = decoder(states_t, z_q)  # (1, h, action_dim)
        actions = torch.argmax(action_logits, dim=-1).squeeze(0)  # (h,)

    return actions.cpu().numpy()


def compute_base_return(
    env: TradingEnv,
    horizon_idx: int,
    base_actions: np.ndarray,
) -> float:
    """使用 base actions 在 env 中执行，返回 horizon 总收益 R_base。

    Args:
        env: 交易环境
        horizon_idx: horizon 索引
        base_actions: 基础动作序列 (h,)

    Returns:
        R_base: 基线 horizon 总收益
    """
    env.reset(horizon_idx)
    total_return = 0.0
    for step_idx in range(len(base_actions)):
        action = int(base_actions[step_idx])
        _, reward, done, _ = env.step(action)
        total_return += reward
        if done:
            break
    return total_return


def run_horizon_with_refinement(
    env: TradingEnv,
    horizon_idx: int,
    base_actions: np.ndarray,
    refinement_agent: RefinementAgent,
    policy_adapter: PolicyAdapter,
    e_a_sel: np.ndarray,
    device: torch.device,
):
    """在一个 horizon 内执行 step 级别的精炼训练。

    # Section 4.3: Step-level policy adapter
    # 每个 step: refinement agent 观测 state + context → 输出 a_ref
    #            → policy adapter 计算 final action → env step
    # 收集 (log_prob, value, reward) 用于 Actor-Critic 更新
    #
    # R_arche_τ = Σ_{i=t}^{τ} r_i^step 逐步累积实时收益（论文定义）
    # τ_remain = t + h - τ 剩余绝对步数（论文定义）

    Args:
        env: 交易环境
        horizon_idx: horizon 索引
        base_actions: 基础动作序列 (h,)
        refinement_agent: 精炼 agent
        policy_adapter: 策略适配器
        e_a_sel: 选定原型的嵌入向量 (code_dim,)
        device: 计算设备

    Returns:
        actual_return: 精炼后的完整 horizon 总收益 R
        log_probs: 每步的 log_prob 列表（仅 RL episode 步）
        values: 每步的 value 列表（仅 RL episode 步）
        a_refs: 每步的 a_ref 列表（仅 RL episode 步）
        adjusted_step: 实际生效调整的步索引，-1 表示无调整
    """
    state = env.reset(horizon_idx)

    h = len(base_actions)
    has_adjusted = False
    adjusted_step = -1  # 实际生效调整的步索引，-1 表示无调整
    actual_return = 0.0
    cumulative_reward = 0.0  # R_arche: 逐步累积的实时收益
    log_probs = []
    values = []
    a_refs = []

    a_base_prev = int(base_actions[0])  # 初始 a_base_prev 设为第一步的 base action

    for step_idx in range(h):
        a_base = int(base_actions[step_idx])

        # Section 4.3: 构建 s_ref1 (市场观测) 和 s_ref2 (上下文)
        # s_ref1 = market observation (state_dim=45)
        s_ref1 = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)  # (1, 45)

        # s_ref2 = [e_a_sel (code_dim=16), a_base (1), R_arche (1), τ_remain (1)]
        # τ_remain = t + h - τ（论文定义：horizon 内剩余的绝对步数）
        tau_remain = float(h - step_idx)
        # R_arche_τ = Σ_{i=t}^{τ} r_i^step（逐步累积的实时收益）
        context = np.concatenate([
            e_a_sel,
            np.array([a_base], dtype=np.float32),
            np.array([cumulative_reward], dtype=np.float32),
            np.array([tau_remain], dtype=np.float32),
        ])  # (code_dim + 3,) = (19,)
        s_ref2 = torch.tensor(
            context, dtype=torch.float32, device=device
        ).unsqueeze(0)  # (1, 19)

        # Section 4.3: Refinement agent 输出调整信号
        action_probs, value = refinement_agent(s_ref1, s_ref2)
        # action_probs: (1, 3), value: (1, 1)

        # 从策略分布中采样 a_ref
        dist = torch.distributions.Categorical(action_probs)
        a_ref_idx = dist.sample()  # (1,) 索引 0/1/2
        log_prob = dist.log_prob(a_ref_idx)  # (1,)

        # 映射索引到 a_ref ∈ {-1, 0, 1}
        a_ref = a_ref_idx.item() - 1  # 0→-1, 1→0, 2→1

        # Section 4.3 / Eq. 6: Policy adapter 计算最终动作
        prev_has_adjusted = has_adjusted
        a_final, has_adjusted = policy_adapter.compute_final_action(a_base, a_base_prev, a_ref, has_adjusted)

        # 在 env 中执行最终动作
        next_state, reward, done, _ = env.step(a_final)
        actual_return += reward
        cumulative_reward += reward  # 更新 R_arche: 逐步累积实时收益

        # 收集训练数据
        log_probs.append(log_prob)
        values.append(value.squeeze())
        a_refs.append(a_ref)

        # 更新状态
        state = next_state
        a_base_prev = a_base

        if done:
            break

        # 论文: "the RL episode terminates as soon as the adapter chooses
        # a non-zero action"
        # RL episode 终止，但仍需执行剩余 base actions 以计算完整
        # horizon return R（用于 Eq.8 的 regret reward）
        if has_adjusted and not prev_has_adjusted:
            adjusted_step = step_idx
            for remaining_idx in range(step_idx + 1, h):
                a_remaining = int(base_actions[remaining_idx])
                _, reward_remaining, done_remaining, _ = env.step(a_remaining)
                actual_return += reward_remaining
                if done_remaining:
                    break
            break

    return actual_return, log_probs, values, a_refs, adjusted_step


def main() -> None:
    # ----------------------------------------------------------------
    # Step 0: 解析配置
    # ----------------------------------------------------------------
    config = parse_args()
    pair = config.pairs[0]  # 单交易对训练
    beta1 = config.refinement_beta1
    beta2 = config.refinement_beta2
    logger.info("Phase III 训练开始: pair=%s", pair)
    logger.info(
        "超参数: total_steps=%d, lr=%.1e, beta1=%.2f, beta2=%.2f, "
        "num_archetypes=%d, discount_factor=%.2f",
        config.phase3_total_steps,
        config.learning_rate,
        beta1,
        beta2,
        config.num_archetypes,
        config.discount_factor,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    # ----------------------------------------------------------------
    # Step 1: 加载 Phase I 和 Phase II 模型
    # 需求 7.4: 检查文件存在性
    # ----------------------------------------------------------------
    codebook, decoder = load_phase1_model(config, pair, device)
    selection_agent = load_phase2_model(config, pair, device)

    # ----------------------------------------------------------------
    # Step 2: 加载特征数据，初始化 TradingEnv
    # ----------------------------------------------------------------
    logger.info("加载特征数据: data_dir=%s, pair=%s", config.data_dir, pair)
    pipeline = FeaturePipeline(config.data_dir, pair)
    train_df, _, _ = pipeline.get_state_vector()
    train_prices_df, _, _ = pipeline.get_prices()

    train_states = train_df.to_numpy()
    train_prices = train_prices_df["close"].to_numpy()

    logger.info(
        "训练集: states shape=%s, prices shape=%s",
        train_states.shape,
        train_prices.shape,
    )

    train_env = TradingEnv(
        states=train_states,
        prices=train_prices,
        pair=pair,
        horizon=config.horizon,
        states_dataframe=train_df,
    )
    logger.info("TradingEnv 初始化完成: train_horizons=%d", train_env.num_horizons)

    if train_env.num_horizons == 0:
        logger.error("训练集 horizon 数量为 0，无法训练")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Step 3: 初始化 RefinementAgent、PolicyAdapter
    # Section 4.3: Step-level policy adapter
    # s_ref2 = [e_a_sel (code_dim=16), a_base (1), R_arche (1), τ_remain (1)] = 19 dim
    # ----------------------------------------------------------------
    context_dim = config.latent_dim + 3  # code_dim + a_base + R_arche + τ_remain = 19

    refinement_agent = RefinementAgent(
        market_dim=config.state_dim,
        context_dim=context_dim,
    ).to(device)

    logger.info(
        "RefinementAgent 初始化完成: params=%d, market_dim=%d, context_dim=%d",
        sum(p.numel() for p in refinement_agent.parameters()),
        config.state_dim,
        context_dim,
    )

    optimizer = torch.optim.Adam(
        refinement_agent.parameters(), lr=config.learning_rate
    )

    # ----------------------------------------------------------------
    # Step 4: 训练循环 — 1M 步（step 级别 RL）
    # Section 4.3: Regret-aware reward
    # r_ref = (R - R_base) + β_1 × (R - R_1_opt)  if a_ref != 0
    # r_ref = 0                                      if a_ref == 0
    # Actor-Critic 更新: advantage = r_ref - V(s)
    # ----------------------------------------------------------------
    gamma = config.discount_factor
    total_steps = config.phase3_total_steps
    log_interval = 100  # 每 100 个 horizon 输出日志

    save_dir = os.path.join(config.result_dir, "phase3_archetype_refinement")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"{pair}_refinement_agent_beta{beta1}.pt"
    )

    reward_history = []
    step_count = 0
    horizon_count = 0

    logger.info("开始训练: %d 步", total_steps)

    while step_count < total_steps:
        # 随机选择一个训练 horizon
        h_idx = np.random.randint(0, train_env.num_horizons)
        h = train_env.horizon
        start = h_idx * h
        end = min(start + h, len(train_env.states))
        horizon_states = train_env.states[start:end]
        horizon_prices = train_env.prices[start:end]

        # Section 4.2: 冻结 SelectionAgent 选择原型
        state_0 = train_env.states[start]
        state_0_t = torch.tensor(
            state_0, dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            action_probs, _ = selection_agent(state_0_t)
            k = torch.argmax(action_probs, dim=-1).item()

        # 获取选定原型的量化嵌入
        e_a_sel_t = codebook.embeddings.weight[k]  # (code_dim,)
        z_q = e_a_sel_t.unsqueeze(0)  # (1, code_dim)
        e_a_sel = e_a_sel_t.detach().cpu().numpy()  # (code_dim,)

        # Section 4.2: 冻结 Decoder 生成 base actions
        base_actions = generate_base_actions(
            decoder, z_q, horizon_states, device
        )  # (h,)

        # 计算 R_base: 使用 base actions 的 horizon 总收益
        R_base = compute_base_return(train_env, h_idx, base_actions)

        # Section 4.3: 在 horizon 内执行 step 级别精炼
        policy_adapter = PolicyAdapter()
        (
            R_actual,
            log_probs,
            values,
            a_refs,
            adjusted_step,
        ) = run_horizon_with_refinement(
            env=train_env,
            horizon_idx=h_idx,
            base_actions=base_actions,
            refinement_agent=refinement_agent,
            policy_adapter=policy_adapter,
            e_a_sel=e_a_sel,
            device=device,
        )

        # Section 4.3 / Eq. 7: 计算 top-5 hindsight-optimal adaptations
        # O_top5 = {(τ_opt^n, a_opt^n, R_opt^n)}_{n=1}^{5}
        top5 = compute_top5_hindsight_optimal(
            prices=horizon_prices,
            base_actions=base_actions,
            step_idx=0,
            env=train_env,
            states=horizon_states,
        )
        R_1_opt = top5[0][2] if top5 else R_base  # top5 现在是 (τ_opt, a_opt, R_opt) 三元组

        # Eq.(9): 构建 hindsight-optimal 动作序列 â_ref
        # â_ref_τ = a_opt^n if τ = τ_opt^n, else 0 (不调整)
        # 索引映射: a_ref_idx 0→a_ref=-1, 1→a_ref=0, 2→a_ref=1
        # 默认 index=1 对应 a_ref=0 (不调整)
        h_actual = len(a_refs)
        optimal_actions = np.ones(h_actual, dtype=np.int64)  # 默认 index=1 (a_ref=0, 不调整)
        if top5:
            # 使用 top-5 的 (τ_opt, a_opt) 构建监督信号
            # top5 按 return 降序排列，同一 τ 只取最优的
            for tau_opt, a_opt, _ in top5:
                if tau_opt < h_actual and optimal_actions[tau_opt] == 1:
                    # a_opt ∈ {-1, 1} → 索引 {0, 2}
                    optimal_actions[tau_opt] = a_opt + 1  # -1→0, 1→2

        # Eq.(8): regret-aware reward 仅在实际生效的调整步赋值
        # 其余步（包括被 Eq.6 阻止的非零 a_ref）统一为 0
        step_rewards = []
        for idx in range(len(a_refs)):
            if idx == adjusted_step:
                r_ref = compute_regret_reward(
                    R=R_actual,
                    R_base=R_base,
                    R_1_opt=R_1_opt,
                    a_ref=a_refs[idx],
                    beta1=beta1,
                )
            else:
                r_ref = 0.0
            step_rewards.append(r_ref)

        # Section 4.3 / Eq. 9: Actor-Critic 更新
        # J' = E[Σ γ^τ r_ref - β_2 × L(â_ref, π_ref)]
        # 计算折扣回报 (从后向前)
        returns = []
        G = 0.0
        for r in reversed(step_rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        # 堆叠 log_probs 和 values
        if log_probs:
            log_probs_t = torch.stack(log_probs)  # (h,)
            values_t = torch.stack(values)  # (h,)

            # Advantage = G_t - V(s_t)
            advantages = returns_t - values_t.detach()

            # Policy loss = -log π(a|s) × advantage
            policy_loss = -(log_probs_t * advantages).mean()

            # Value loss = (G_t - V(s_t))²
            value_loss = F.mse_loss(values_t, returns_t)

            # Eq. 9: β_2 × L(â_ref, π_ref) — 交叉熵监督损失
            # 使用 hindsight-optimal 动作 â_ref 作为监督标签
            # 重新前向传播获取 action logits（用于交叉熵计算）
            optimal_actions_t = torch.tensor(
                optimal_actions[:h_actual], dtype=torch.long, device=device
            )
            # 收集每步的 action_probs 用于交叉熵
            # 重新前向传播以获取 logits
            ce_loss = torch.tensor(0.0, device=device)
            state_replay = train_env.reset(h_idx)
            has_adjusted_replay = False
            a_base_prev_replay = int(base_actions[0])
            cumulative_reward_replay = 0.0

            for step_idx in range(h_actual):
                a_base_r = int(base_actions[step_idx])
                s_ref1_r = torch.tensor(
                    state_replay, dtype=torch.float32, device=device
                ).unsqueeze(0)
                tau_remain_r = float(h - step_idx)
                context_r = np.concatenate([
                    e_a_sel,
                    np.array([a_base_r], dtype=np.float32),
                    np.array([cumulative_reward_replay], dtype=np.float32),
                    np.array([tau_remain_r], dtype=np.float32),
                ])
                s_ref2_r = torch.tensor(
                    context_r, dtype=torch.float32, device=device
                ).unsqueeze(0)

                action_probs_r, _ = refinement_agent(s_ref1_r, s_ref2_r)
                # 交叉熵: -log π(â_ref | s_ref)
                target_r = optimal_actions_t[step_idx].unsqueeze(0)
                ce_step = F.cross_entropy(
                    torch.log(action_probs_r + 1e-8), target_r
                )
                ce_loss = ce_loss + ce_step

                # 重放环境步骤以获取正确的下一状态
                a_ref_r = a_refs[step_idx]
                a_final_r, has_adjusted_replay = policy_adapter.compute_final_action(
                    a_base_r, a_base_prev_replay, a_ref_r, has_adjusted_replay
                )
                next_state_r, reward_r, done_r, _ = train_env.step(a_final_r)
                cumulative_reward_replay += reward_r
                state_replay = next_state_r
                a_base_prev_replay = a_base_r
                if done_r:
                    break

            ce_loss = ce_loss / h_actual

            # Eq. 9: 总损失 = policy_loss + value_loss + β_2 × cross_entropy_loss
            loss = policy_loss + value_loss + beta2 * ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 更新计数器
        step_count += len(a_refs)
        horizon_count += 1
        reward_history.append(R_actual)

        # 日志输出
        if horizon_count % log_interval == 0:
            recent_rewards = reward_history[-log_interval:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            logger.info(
                "Step %7d/%d (horizon %d) — avg_reward=%.4f, R=%.4f, "
                "R_base=%.4f, R_1_opt=%.4f, loss=%.4f",
                step_count,
                total_steps,
                horizon_count,
                avg_reward,
                R_actual,
                R_base,
                R_1_opt,
                loss.item() if log_probs else 0.0,
            )

        # NaN 检测
        if log_probs and np.isnan(loss.item()):
            logger.error(
                "训练 loss 发散 (NaN)，在 step %d 终止训练", step_count
            )
            break

    # ----------------------------------------------------------------
    # Step 5: 保存模型到 result/phase3_archetype_refinement/
    # ----------------------------------------------------------------
    torch.save(
        {
            "agent": refinement_agent.state_dict(),
            "training_rewards": reward_history,
            "step": step_count,
            "beta1": beta1,
            "config": {
                "state_dim": config.state_dim,
                "latent_dim": config.latent_dim,
                "num_archetypes": config.num_archetypes,
                "phase3_total_steps": config.phase3_total_steps,
                "refinement_beta1": beta1,
                "refinement_beta2": beta2,
                "learning_rate": config.learning_rate,
                "discount_factor": config.discount_factor,
            },
        },
        save_path,
    )
    logger.info("模型已保存到 %s", save_path)

    # ----------------------------------------------------------------
    # Step 6: 输出训练日志摘要
    # ----------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("Phase III 训练完成: pair=%s, beta1=%.2f", pair, beta1)
    logger.info("总训练步数: %d, 总 horizon 数: %d", step_count, horizon_count)
    logger.info(
        "最终平均奖励 (最近 1000 horizons): %.4f",
        np.mean(reward_history[-1000:]) if reward_history else float("nan"),
    )
    logger.info("模型保存路径: %s", save_path)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
