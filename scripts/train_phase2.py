#!/usr/bin/env python
"""Phase II 训练脚本 — 原型选择

# 需求: 7.2, 5.3, 5.4, 5.5, 5.7, 7.4, 7.5, 7.6, 7.7
#
# 流程:
# 1. 加载 Phase I 模型（码本 + 冻结 Decoder），检查文件存在性
# 2. 加载特征数据，初始化 TradingEnv（训练集 + 验证集）
# 3. 初始化 SelectionAgent（Actor-Critic）
# 4. 训练 3M 步（horizon 级别 RL）
#    - 每个 horizon: agent 选择原型 → 冻结 decoder 生成 micro actions → env 执行 → 计算 horizon return
#    - Actor-Critic 更新: advantage = R - V(s)
#    - KL 惩罚: α × KL(π || uniform)，α=1
#    - Policy loss = -log π(k|s) × advantage + α × KL
#    - Value loss = (R - V(s))²
# 5. 定期在验证集上评估，保存最优检查点
# 6. 保存模型到 result/phase2_archetype_selection/
#
# 用法:
#   python scripts/train_phase2.py --pair BTC
#   python scripts/train_phase2.py --pair ETH --phase2-total-steps 1000000 --lr 1e-4
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
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_phase1_model(config, pair: str, device: torch.device):
    """加载 Phase I 模型（码本 + 冻结 Decoder）。

    # 需求 7.4: 前置阶段模型文件不存在时抛出明确错误
    # 需求 5.3: 冻结 Decoder 参数

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

    # 需求 5.3: 冻结 Codebook 和 Decoder — 不参与梯度更新
    for param in codebook.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    codebook.eval()
    decoder.eval()

    logger.info("Phase I 模型加载完成，Codebook 和 Decoder 已冻结")
    return codebook, decoder



def run_horizon_with_decoder(
    env: TradingEnv,
    horizon_idx: int,
    decoder: VQDecoder,
    z_q: torch.Tensor,
    device: torch.device,
) -> float:
    """使用冻结 Decoder 在一个 horizon 内执行交易，返回 horizon 总收益。

    # Section 4.2: 冻结 Decoder 生成 micro actions
    # 1. 收集 horizon 内所有状态
    # 2. Decoder 根据状态和 z_q 生成 action logits
    # 3. argmax 得到 micro actions
    # 4. 在 env 中逐步执行，累计 horizon return

    Args:
        env: 交易环境
        horizon_idx: horizon 索引
        decoder: 冻结的 VQ Decoder
        z_q: 选定原型的量化嵌入 (1, code_dim)
        device: 计算设备

    Returns:
        horizon_return: 该 horizon 的总收益（未折扣）
    """
    state = env.reset(horizon_idx)
    horizon_return = 0.0

    # 收集 horizon 内所有状态用于 decoder 批量推理
    h = env.horizon
    start = horizon_idx * h
    end = min(start + h, len(env.states))
    horizon_states = env.states[start:end]  # (h, state_dim)

    # Decoder 批量生成 action logits
    states_t = torch.tensor(horizon_states, dtype=torch.float32, device=device).unsqueeze(0)
    # states_t: (1, h, state_dim)

    with torch.no_grad():
        action_logits = decoder(states_t, z_q)  # (1, h, action_dim)
        actions = torch.argmax(action_logits, dim=-1).squeeze(0)  # (h,)
        actions_np = actions.cpu().numpy()

    # 在 env 中逐步执行 micro actions
    for step_idx in range(len(actions_np)):
        action = int(actions_np[step_idx])
        _, reward, done, _ = env.step(action)
        horizon_return += reward
        if done:
            break

    return horizon_return


def evaluate_on_validation(
    agent: SelectionAgent,
    codebook: VQCodebook,
    decoder: VQDecoder,
    val_env: TradingEnv,
    device: torch.device,
) -> float:
    """在验证集上评估 SelectionAgent，返回平均 horizon return。

    # 需求 5.7: 定期在验证集上评估性能

    Args:
        agent: SelectionAgent
        codebook: 冻结的码本
        decoder: 冻结的 Decoder
        val_env: 验证集环境
        device: 计算设备

    Returns:
        平均 horizon return
    """
    agent.eval()
    total_return = 0.0
    num_horizons = val_env.num_horizons

    if num_horizons == 0:
        return 0.0

    for h_idx in range(num_horizons):
        state = val_env.states[h_idx * val_env.horizon]
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_probs, _ = agent(state_t)
            k = torch.argmax(action_probs, dim=-1).item()

        # 获取选定原型的量化嵌入
        z_q = codebook.embeddings.weight[k].unsqueeze(0)  # (1, code_dim)

        horizon_ret = run_horizon_with_decoder(val_env, h_idx, decoder, z_q, device)
        total_return += horizon_ret

    agent.train()
    return total_return / num_horizons



def main() -> None:
    # ----------------------------------------------------------------
    # Step 0: 解析配置
    # ----------------------------------------------------------------
    config = parse_args()
    pair = config.pairs[0]  # 单交易对训练
    logger.info("Phase II 训练开始: pair=%s", pair)
    logger.info(
        "超参数: total_steps=%d, lr=%.1e, selection_alpha=%.2f, "
        "num_archetypes=%d, discount_factor=%.2f",
        config.phase2_total_steps,
        config.learning_rate,
        config.selection_alpha,
        config.num_archetypes,
        config.discount_factor,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    # ----------------------------------------------------------------
    # Step 1: 加载 Phase I 模型（码本 + 冻结 Decoder）
    # ----------------------------------------------------------------
    codebook, decoder = load_phase1_model(config, pair, device)

    # ----------------------------------------------------------------
    # Step 2: 加载特征数据，初始化 TradingEnv
    # ----------------------------------------------------------------
    logger.info("加载特征数据: data_dir=%s, pair=%s", config.data_dir, pair)
    pipeline = FeaturePipeline(config.data_dir, pair, config)
    states = pipeline.get_state_vector()  # (T, 45)

    # 使用第一列作为价格代理
    # [NOTE: 论文未明确指定价格列，使用 states 第 0 列作为价格]
    prices = states[:, 0].copy()

    # 按时间划分
    train_states, val_states, _ = pipeline.split_by_date(states)
    train_prices = prices[: len(train_states)]
    val_prices = prices[len(train_states) : len(train_states) + len(val_states)]

    logger.info(
        "训练集: states shape=%s, 验证集: states shape=%s",
        train_states.shape,
        val_states.shape,
    )

    train_env = TradingEnv(
        states=train_states,
        prices=train_prices,
        pair=pair,
        horizon=config.horizon,
    )
    val_env = TradingEnv(
        states=val_states,
        prices=val_prices,
        pair=pair,
        horizon=config.horizon,
    )
    logger.info(
        "TradingEnv 初始化完成: train_horizons=%d, val_horizons=%d",
        train_env.num_horizons,
        val_env.num_horizons,
    )

    if train_env.num_horizons == 0:
        logger.error("训练集 horizon 数量为 0，无法训练")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Step 3: 初始化 SelectionAgent
    # ----------------------------------------------------------------
    agent = SelectionAgent(
        state_dim=config.state_dim,
        num_archetypes=config.num_archetypes,
    ).to(device)

    logger.info(
        "SelectionAgent 初始化完成: params=%d",
        sum(p.numel() for p in agent.parameters()),
    )

    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    # ----------------------------------------------------------------
    # Step 4: 训练循环 — 3M 步（horizon 级别 RL）
    # Section 4.2: Horizon-level RL
    # 目标函数含 KL 惩罚: α × KL(π || uniform), α=1
    # Policy loss = -log π(k|s) × advantage + α × KL
    # Value loss = (R - V(s))²
    # ----------------------------------------------------------------
    alpha = config.selection_alpha  # KL 惩罚系数，默认 1.0
    gamma = config.discount_factor
    K = config.num_archetypes
    uniform_prob = 1.0 / K  # 均匀分布概率

    total_steps = config.phase2_total_steps
    val_interval = max(train_env.num_horizons, 1000)  # 每遍历一次训练集或 1000 步评估一次
    log_interval = 100  # 每 100 步输出日志

    best_val_return = float("-inf")
    save_dir = os.path.join(config.result_dir, "phase2_archetype_selection")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{pair}_selection_agent.pt")

    reward_history = []
    step_count = 0

    logger.info("开始训练: %d 步", total_steps)

    while step_count < total_steps:
        # 随机选择一个训练 horizon
        h_idx = np.random.randint(0, train_env.num_horizons)

        # 获取 horizon 起始状态
        state = train_env.states[h_idx * train_env.horizon]
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # state_t: (1, state_dim)

        # Section 4.2: Agent 选择原型
        action_probs, value = agent(state_t)
        # action_probs: (1, K), value: (1, 1)

        # 从策略分布中采样原型索引
        dist = torch.distributions.Categorical(action_probs)
        k = dist.sample()  # (1,)
        log_prob = dist.log_prob(k)  # (1,)

        # 获取选定原型的量化嵌入
        z_q = codebook.embeddings.weight[k.item()].unsqueeze(0)  # (1, code_dim)

        # Section 4.2: 冻结 Decoder 生成 micro actions → env 执行 → horizon return
        horizon_return = run_horizon_with_decoder(
            train_env, h_idx, decoder, z_q, device
        )

        reward_history.append(horizon_return)

        # Section 4.2: Actor-Critic 更新
        R = torch.tensor([horizon_return], dtype=torch.float32, device=device)
        advantage = R - value.squeeze()  # advantage = R - V(s)

        # KL 惩罚: KL(π || uniform) = Σ π(k) × log(π(k) / (1/K))
        # = Σ π(k) × (log π(k) + log K)
        kl_divergence = torch.sum(
            action_probs * (torch.log(action_probs + 1e-8) - np.log(uniform_prob)),
            dim=-1,
        )  # (1,)

        # Policy loss = -log π(k|s) × advantage.detach() + α × KL
        policy_loss = -log_prob * advantage.detach() + alpha * kl_divergence

        # Value loss = (R - V(s))²
        value_loss = advantage.pow(2)

        # 总损失
        loss = (policy_loss + value_loss).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_count += 1

        # 日志输出
        if step_count % log_interval == 0:
            recent_rewards = reward_history[-log_interval:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            logger.info(
                "Step %7d/%d — avg_reward=%.4f, loss=%.4f, kl=%.4f",
                step_count,
                total_steps,
                avg_reward,
                loss.item(),
                kl_divergence.item(),
            )

        # 需求 5.7: 定期在验证集上评估，保存最优检查点
        if step_count % val_interval == 0:
            val_return = evaluate_on_validation(
                agent, codebook, decoder, val_env, device
            )
            logger.info(
                "验证集评估 (step %d): avg_return=%.4f (best=%.4f)",
                step_count,
                val_return,
                best_val_return,
            )

            if val_return > best_val_return:
                best_val_return = val_return
                torch.save(
                    {
                        "agent": agent.state_dict(),
                        "training_rewards": reward_history,
                        "best_validation_return": best_val_return,
                        "step": step_count,
                        "config": {
                            "state_dim": config.state_dim,
                            "num_archetypes": config.num_archetypes,
                            "selection_alpha": config.selection_alpha,
                            "phase2_total_steps": config.phase2_total_steps,
                            "learning_rate": config.learning_rate,
                            "discount_factor": config.discount_factor,
                        },
                    },
                    save_path,
                )
                logger.info("最优模型已保存到 %s (val_return=%.4f)", save_path, val_return)

    # ----------------------------------------------------------------
    # Step 5: 最终保存（如果训练结束时不是最优也保存最终版本）
    # ----------------------------------------------------------------
    final_save_path = os.path.join(save_dir, f"{pair}_selection_agent_final.pt")
    torch.save(
        {
            "agent": agent.state_dict(),
            "training_rewards": reward_history,
            "best_validation_return": best_val_return,
            "step": step_count,
            "config": {
                "state_dim": config.state_dim,
                "num_archetypes": config.num_archetypes,
                "selection_alpha": config.selection_alpha,
                "phase2_total_steps": config.phase2_total_steps,
                "learning_rate": config.learning_rate,
                "discount_factor": config.discount_factor,
            },
        },
        final_save_path,
    )
    logger.info("最终模型已保存到 %s", final_save_path)

    # ----------------------------------------------------------------
    # Step 6: 输出训练日志摘要
    # ----------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("Phase II 训练完成: pair=%s", pair)
    logger.info("总训练步数: %d", step_count)
    logger.info(
        "最终平均奖励 (最近 1000 步): %.4f",
        np.mean(reward_history[-1000:]) if reward_history else float("nan"),
    )
    logger.info("最优验证集 return: %.4f", best_val_return)
    logger.info("最优模型路径: %s", save_path)
    logger.info("最终模型路径: %s", final_save_path)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
