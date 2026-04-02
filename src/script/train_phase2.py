#!/usr/bin/env python
"""Phase II 训练脚本 — 原型选择（PPO 风格）

# 需求: 7.2, 5.3, 5.4, 5.5, 5.7, 7.4, 7.5, 7.6, 7.7
#
# 流程:
# 1. 加载 Phase I 模型（码本 + 冻结 Decoder），检查文件存在性
# 2. 加载特征数据，初始化 TradingEnv（训练集 + 验证集）
# 3. 初始化 SelectionAgent（Actor-Critic backbone）
# 4. 训练 3M 步（horizon 级别 RL / PPO 风格）
#    - 每个 horizon: agent 选择原型 → 冻结 decoder 生成 micro actions → env 执行 → 计算 horizon return
#    - PPO 更新: clipped surrogate objective + value loss + entropy bonus
#    - imitation / KL 惩罚: α × KL(â_sel || π_sel)
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

import os
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
from src.phase2.selection_agent import SelectionAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _parameter_grad_norm(parameters) -> float:
    """计算一组参数当前梯度的 L2 norm。

    功能说明:
        用于观察 PPO 更新时 policy / value 头是否真的在收到梯度，
        便于排查“critic 压过 actor”或“policy 基本没学动”的问题。
    """
    total_sq = 0.0
    has_grad = False
    for param in parameters:
        if param.grad is None:
            continue
        grad_norm = float(param.grad.detach().data.norm(2).item())
        total_sq += grad_norm * grad_norm
        has_grad = True
    if not has_grad:
        return 0.0
    return float(total_sq ** 0.5)


def _histogram_counts(values: np.ndarray | list[int], num_bins: int) -> np.ndarray:
    """把离散标签序列转成固定长度直方图计数。"""
    values_np = np.asarray(values, dtype=np.int64).reshape(-1)
    if values_np.size == 0:
        return np.zeros(num_bins, dtype=np.int64)
    valid = values_np[(values_np >= 0) & (values_np < num_bins)]
    if valid.size == 0:
        return np.zeros(num_bins, dtype=np.int64)
    return np.bincount(valid, minlength=num_bins).astype(np.int64)


def _format_histogram_from_counts(counts: np.ndarray | list[int]) -> str:
    """把直方图计数格式化成紧凑日志字符串。"""
    counts_np = np.asarray(counts, dtype=np.int64).reshape(-1)
    return "[" + ", ".join(f"{idx}:{int(v)}" for idx, v in enumerate(counts_np.tolist())) + "]"


def _aggregate_execution_diagnostics(horizon_details: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总一批 horizon 执行诊断指标。

    功能说明:
        把 decoder 在环境中的 horizon 级执行结果拆成 gross pnl / cost / turnover
        / 换仓次数等指标，避免只看最终 reward 而无法定位负收益来源。
    """
    if not horizon_details:
        return {
            "avg_return": 0.0,
            "avg_gross_pnl": 0.0,
            "avg_execution_cost": 0.0,
            "avg_commission": 0.0,
            "avg_slippage": 0.0,
            "avg_position_changes": 0.0,
            "avg_direct_flips": 0.0,
            "avg_turnover": 0.0,
            "decoder_action_histogram": _format_histogram_from_counts(np.zeros(3, dtype=np.int64)),
        }

    decoder_hist = np.sum(
        [np.asarray(item["decoder_action_histogram"], dtype=np.int64) for item in horizon_details],
        axis=0,
    )

    return {
        "avg_return": float(np.mean([item["horizon_return"] for item in horizon_details])),
        "avg_gross_pnl": float(np.mean([item["gross_pnl"] for item in horizon_details])),
        "avg_execution_cost": float(np.mean([item["execution_cost_total"] for item in horizon_details])),
        "avg_commission": float(np.mean([item["commission_total"] for item in horizon_details])),
        "avg_slippage": float(np.mean([item["slippage_total"] for item in horizon_details])),
        "avg_position_changes": float(np.mean([item["num_position_changes"] for item in horizon_details])),
        "avg_direct_flips": float(np.mean([item["num_direct_flips"] for item in horizon_details])),
        "avg_turnover": float(np.mean([item["turnover_total"] for item in horizon_details])),
        "decoder_action_histogram": _format_histogram_from_counts(decoder_hist),
    }


def _run_policy_on_horizons(
    codebook: VQCodebook,
    decoder: VQDecoder,
    env: TradingEnv,
    horizon_indices: np.ndarray,
    device: torch.device,
    selected_archetypes: np.ndarray,
) -> dict[str, Any]:
    """在给定 horizons 上执行指定 archetype 选择结果，并汇总诊断指标。"""
    if len(horizon_indices) != len(selected_archetypes):
        raise ValueError(
            f"horizon_indices 和 selected_archetypes 长度不一致: {len(horizon_indices)} vs {len(selected_archetypes)}"
        )

    horizon_details: list[dict[str, Any]] = []
    for h_idx, archetype_idx in zip(horizon_indices.tolist(), selected_archetypes.tolist()):
        z_q = codebook.embeddings.weight[int(archetype_idx)].unsqueeze(0)
        detail = run_horizon_with_decoder(
            env=env,
            horizon_idx=int(h_idx),
            decoder=decoder,
            z_q=z_q,
            device=device,
            return_details=True,
        )
        horizon_details.append(detail)

    metrics = _aggregate_execution_diagnostics(horizon_details)
    metrics["selected_histogram"] = _format_histogram_from_counts(
        _histogram_counts(selected_archetypes, codebook.embeddings.weight.size(0))
    )
    metrics["num_horizons"] = int(len(horizon_indices))
    return metrics


def _cfg(config: Any, name: str, default: Any) -> Any:
    """安全读取配置项；若不存在则回退到默认值。

    功能说明:
        为 PPO 新增超参数提供向后兼容能力；即使 src.config.parse_args
        尚未加入这些字段，本脚本也可以直接运行。

    论文相关:
        论文本身定义了 Phase II 的高层 MDP 和目标函数 Eq.(5)，
        但未强制规定 PPO 的工程超参数；因此这里把 rollout/minibatch/
        clip/entropy 等都做成可选配置，属于训练器层面的实现细节。
    """
    return getattr(config, name, default)


def get_phase2_hparams(config: Any) -> dict[str, Any]:
    """读取 PPO 相关超参数。

    功能说明:
        从 config 中读取 Phase II 的 PPO 风格训练参数，若外部配置未定义，
        则使用安全默认值。
    ppo 参数说明：
        rollout_batch_size	每轮收集的样本数（horizon 数量），用于构建经验池
        ppo_epochs	对同一批数据重复训练的轮数（通常 3-10）
        minibatch_size	每个 epoch 内切分成的小批量大小
        clip_eps	策略裁剪范围（如 0.2 表示新旧策略概率比限制在 [0.8, 1.2]），防止策略突变
        vf_coef	价值函数损失的权重系数（总 loss = policy_loss + vf_coef × value_loss）
        ent_coef	熵正则化系数，鼓励探索（越大越倾向于均匀分布）
        max_grad_norm	梯度裁剪阈值，防止梯度爆炸
        log_interval	每 N 步输出一次日志
        eval_max_horizons	验证集评估时最多评估的 horizon 数量（None 表示全部）
        diagnostic_horizons	训练子集诊断时抽样的 horizon 数量
    论文相关:
        论文的核心是 Section 4.2 的 horizon-level selector 与 Eq.(5) 的目标，
        这里的 clip_eps / ppo_epochs / minibatch_size / ent_coef / vf_coef
        是为了把原先的单步 Actor-Critic 升级为更稳定的 PPO 风格优化器。

    Returns:
        dict[str, Any]: 统一整理后的 PPO 超参数字典。
    """
    rollout_batch_size = int(_cfg(config, "phase2_rollout_batch_size", 1024))
    ppo_epochs = int(_cfg(config, "phase2_ppo_epochs", 4))
    minibatch_size = int(_cfg(config, "phase2_minibatch_size", 256))
    clip_eps = float(_cfg(config, "phase2_clip_eps", 0.2))
    vf_coef = float(_cfg(config, "phase2_vf_coef", 0.2))
    ent_coef = float(_cfg(config, "phase2_ent_coef", 0.02))
    max_grad_norm = float(_cfg(config, "phase2_max_grad_norm", 1.0))
    log_interval = int(_cfg(config, "phase2_log_interval", 100000))
    eval_max_horizons = _cfg(config, "phase2_eval_max_horizons", None)
    diagnostic_horizons = int(_cfg(config, "phase2_diagnostic_horizons", 128))

    rollout_batch_size = max(1, rollout_batch_size)
    ppo_epochs = max(1, ppo_epochs)
    minibatch_size = max(1, minibatch_size)
    log_interval = max(1, log_interval)
    diagnostic_horizons = max(1, diagnostic_horizons)

    # PPO 关键保护：minibatch 必须小于 rollout_batch，否则第一轮 full-batch
    # 更新在 advantage 零均值归一化后很容易导致 policy loss 接近 0。
    if rollout_batch_size > 1 and minibatch_size >= rollout_batch_size:
        adjusted_minibatch = max(1, rollout_batch_size // 4)
        logger.warning(
            "检测到 minibatch_size(%d) >= rollout_batch_size(%d)，自动调整为 %d，避免 full-batch PPO 导致 actor 更新退化。",
            minibatch_size,
            rollout_batch_size,
            adjusted_minibatch,
        )
        minibatch_size = adjusted_minibatch

    return {
        "rollout_batch_size": rollout_batch_size,
        "ppo_epochs": ppo_epochs,
        "minibatch_size": minibatch_size,
        "clip_eps": clip_eps,
        "vf_coef": vf_coef,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "log_interval": log_interval,
        "eval_max_horizons": eval_max_horizons,
        "diagnostic_horizons": diagnostic_horizons,
    }


def load_phase1_model(config: Any, pair: str, device: torch.device):
    """加载 Phase I 模型（编码器 + 码本 + 冻结 Decoder）。

    # 需求 7.4: 前置阶段模型文件不存在时抛出明确错误
    # 需求 5.3: 冻结 Decoder 参数

    功能说明:
        读取 Phase I 训练好的 VQEncoder / VQCodebook / VQDecoder。
        在 Phase II 中，这三部分都作为“已学习好的 archetype prior”使用，
        不再参与梯度更新。

    论文相关:
        - Phase I 对应 Archetype Discovery；
        - Phase II 对应 Archetype Selection；
        - Section 4.2 明确要求：selector 选出 archetype 后，
          由 frozen decoder p_theta_d(a_base | s, e_{a_sel}) 生成 micro actions；
        - 同时，ground-truth archetype label â_sel 由冻结 encoder + codebook
          对当前 horizon 的 demonstration chunk 编码得到，对应论文 Eq.(5)。

    Returns:
        encoder: 加载权重后的 VQEncoder（冻结，用于获取 ground-truth archetype label）
        codebook: 加载权重后的 VQCodebook（冻结）
        decoder: 加载权重后的 VQDecoder（冻结）
    """
    model_path = os.path.join(
        config.result_dir,
        pair,
        "phase1_archetype_discovery",
        f"{pair}_vq_model.pt",
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Phase I 模型文件不存在: {model_path}\n"
            f"请先运行 Phase I 训练: python scripts/train_phase1.py --pair {pair}"
        )

    logger.info("加载 Phase I 模型: %s", model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 初始化并加载 Encoder（用于获取 ground-truth archetype label）
    encoder = VQEncoder(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.lstm_hidden_dim,
        latent_dim=config.latent_dim,
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])

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

    # 需求 5.3: 冻结 Encoder、Codebook 和 Decoder — 不参与梯度更新
    for param in encoder.parameters():
        param.requires_grad = False
    for param in codebook.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    encoder.eval()
    codebook.eval()
    decoder.eval()

    logger.info("Phase I 模型加载完成，Encoder、Codebook 和 Decoder 已冻结")
    return encoder, codebook, decoder


def run_horizon_with_decoder(
    env: TradingEnv,
    horizon_idx: int,
    decoder: VQDecoder,
    z_q: torch.Tensor,
    device: torch.device,
    return_details: bool = False,
) -> float | dict[str, Any]:
    """使用冻结 Decoder 在一个 horizon 内执行交易，返回 horizon 总收益。

    # Section 4.2: 冻结 Decoder 生成 micro actions
    # 1. 收集 horizon 内所有状态
    # 2. Decoder 根据状态和 z_q 生成 action logits
    # 3. argmax 得到 micro actions
    # 4. 在 env 中逐步执行，累计 horizon return

    功能说明:
        该函数负责把“高层 archetype 决策”真正落地为一个 horizon 内的
        低层执行收益：先用 decoder 生成整段 micro action，再逐步喂给 env，
        最终得到该 horizon 的累计回报。

        为了排查 reward 为负的来源，这里额外统计 gross pnl / execution cost /
        turnover / direct flips 等诊断信息；默认仍返回 float，保持原调用方式兼容。

    论文相关:
        - 对应 Section 4.2 中：选定 archetype 后，将其 code e_{a_sel}
          输入 frozen decoder p_theta_d(a_base | s, e_{a_sel})；
        - 该函数输出的 horizon_return 对应论文里的 r_t^sel，
          即一个 horizon 内所有 step reward 的求和。

    Args:
        env: 交易环境
        horizon_idx: horizon 索引
        decoder: 冻结的 VQ Decoder
        z_q: 选定原型的量化嵌入 (1, code_dim)
        device: 计算设备
        return_details: 是否返回执行细分统计

    Returns:
        horizon_return: 该 horizon 的总收益（未折扣）
        或包含收益拆分项的 detail 字典
    """
    # 进入指定 horizon；保留 reset 调用，确保 env 内部游标与当前 horizon 对齐。
    state = env.reset(horizon_idx)

    stats: dict[str, Any] = {
        "horizon_return": 0.0,
        "gross_pnl": 0.0,
        "execution_cost_total": 0.0,
        "commission_total": 0.0,
        "slippage_total": 0.0,
        "num_position_changes": 0,
        "num_direct_flips": 0,
        "turnover_total": 0.0,
        "num_steps": 0,
        "decoder_action_histogram": [0, 0, 0],
    }

    # 收集 horizon 内所有状态用于 decoder 批量推理
    h = env.horizon
    start = horizon_idx * h
    end = min(start + h, len(env.states))
    horizon_states = env.states[start:end]  # (h, state_dim)

    # Decoder 批量生成 action logits
    states_t = torch.tensor(
        horizon_states, dtype=torch.float32, device=device
    ).unsqueeze(0)
    # states_t: (1, h, state_dim)

    with torch.no_grad():
        action_logits = decoder(states_t, z_q)  # (1, h, action_dim)
        actions = torch.argmax(action_logits, dim=-1).squeeze(0)  # (h,)
        actions_np = actions.detach().cpu().numpy()

    # 在 env 中逐步执行 micro actions
    for step_idx in range(len(actions_np)):
        action = int(actions_np[step_idx])
        if 0 <= action < len(stats["decoder_action_histogram"]):
            stats["decoder_action_histogram"][action] += 1

        _, reward, done, info = env.step(action)

        old_position = int(info.get("old_position", 0))
        new_position = int(info.get("position", old_position))
        execution_cost = float(info.get("execution_cost", 0.0))
        price = float(info.get("price", 0.0))
        delta_position = int(new_position - old_position)

        commission = float(env.COMMISSION_RATE * abs(delta_position) * price)
        commission = min(commission, execution_cost)
        slippage = max(0.0, execution_cost - commission)
        gross_pnl = float(reward + execution_cost)

        if old_position != new_position:
            stats["num_position_changes"] += 1
        if old_position != 0 and new_position != 0 and np.sign(old_position) != np.sign(new_position):
            stats["num_direct_flips"] += 1

        stats["turnover_total"] += float(abs(delta_position))
        stats["horizon_return"] += float(reward)
        stats["gross_pnl"] += gross_pnl
        stats["execution_cost_total"] += execution_cost
        stats["commission_total"] += commission
        stats["slippage_total"] += slippage
        stats["num_steps"] += 1

        if done:
            break

    # state 变量仅用于保留 reset 语义和调试语境；逻辑上无需额外使用。
    _ = state
    if return_details:
        return stats
    return float(stats["horizon_return"])


def get_horizon_start_states(env: TradingEnv, horizon_indices: np.ndarray) -> np.ndarray:
    """获取一批 horizon 的起始状态。

    功能说明:
        将一组 horizon index 映射到对应的首 bar 状态，用于构建 selector 的
        batch 输入。

    论文相关:
        Section 4.2 明确规定高层状态 s_sel 定义为当前 horizon 第一根 bar 的
        状态向量，因此这里严格按 horizon 起点取状态，而不是取整段序列。
    """
    start_indices = horizon_indices * env.horizon
    return env.states[start_indices]


def get_ground_truth_labels(
    encoder: VQEncoder,
    codebook: VQCodebook,
    demo_states: np.ndarray,
    demo_actions: np.ndarray,
    demo_rewards: np.ndarray,
    horizon_indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """批量计算 Eq.(5) 中的 ground-truth archetype label。

    功能说明:
        从 DP demonstration dataset 中取出指定 horizon 的示范轨迹，
        通过冻结的 VQEncoder + VQCodebook 计算离散 archetype index，
        作为当前 horizon 的监督标签。

    论文相关:
        - 这里得到的 gt label 就是论文 Eq.(5) 中的 â_sel；
        - 其作用不是强行替代 RL，而是作为 KL regularization / imitation prior，
          让高层 selector 在探索时仍保持与 demonstration archetype 的一致性。

    Returns:
        torch.Tensor: 形状为 (batch,) 的 ground-truth archetype index。
    """
    demo_s = torch.tensor(demo_states[horizon_indices], dtype=torch.float32, device=device)
    demo_a = torch.tensor(demo_actions[horizon_indices], dtype=torch.long, device=device)
    demo_r = torch.tensor(demo_rewards[horizon_indices], dtype=torch.float32, device=device)

    with torch.no_grad():
        z_e = encoder(demo_s, demo_a, demo_r)  # (batch, latent_dim)
        _, gt_indices, _ = codebook.quantize(z_e)  # (batch,)

    return gt_indices.long()


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

    实现说明:
        advantage 这里采用简化的一步形式 advantage = return - value，
        更接近原始代码结构；虽然不是全量 GAE，但已经满足 PPO 风格更新所需。

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

    returns: list[float] = []
    rollout_details: list[dict[str, Any]] = []
    for i, h_idx in enumerate(horizon_indices):
        # 获取选定原型的量化嵌入；对应论文中的 e_{a_sel}。
        z_q = codebook.embeddings.weight[actions[i].item()].unsqueeze(0)  # (1, code_dim)

        # Section 4.2: 冻结 Decoder 生成 micro actions → env 执行 → horizon return
        # horizon return 就是这一个 horizon 的奖励总和。
        detail = run_horizon_with_decoder(
            train_env, int(h_idx), decoder, z_q, device, return_details=True
        )
        returns.append(float(detail["horizon_return"]))
        rollout_details.append(detail)

    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
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

    actions_np = actions.detach().cpu().numpy()
    greedy_np = greedy_actions.detach().cpu().numpy()
    gt_np = gt_labels.detach().cpu().numpy()

    diagnostics = _aggregate_execution_diagnostics(rollout_details)
    diagnostics.update(
        {
            "raw_adv_mean": float(raw_advantages_t.mean().item()) if raw_advantages_t.numel() > 0 else 0.0,
            "raw_adv_std": float(raw_advantages_t.std(unbiased=False).item()) if raw_advantages_t.numel() > 0 else 0.0,
            "sampled_archetype_histogram": _format_histogram_from_counts(
                _histogram_counts(actions_np, agent.num_archetypes)
            ),
            "greedy_archetype_histogram": _format_histogram_from_counts(
                _histogram_counts(greedy_np, agent.num_archetypes)
            ),
            "gt_label_histogram": _format_histogram_from_counts(
                _histogram_counts(gt_np, agent.num_archetypes)
            ),
            "sampled_gt_agreement": float(np.mean(actions_np == gt_np)) if gt_np.size > 0 else 0.0,
            "greedy_gt_agreement": float(np.mean(greedy_np == gt_np)) if gt_np.size > 0 else 0.0,
        }
    )

    return {
        "states": states_t,
        "actions": actions.detach(),
        "old_log_probs": old_log_probs.detach(),
        "returns": returns_t.detach(),
        "advantages": advantages_t.detach(),
        "gt_labels": gt_labels.detach(),
        "diagnostics": diagnostics,
    }


def ppo_update(
    agent: SelectionAgent,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    alpha: float,
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

            # Eq.(5): KL(â_sel || π_sel)
            # one-hot(gt) 对 policy 的 KL，可等价实现为 NLL / cross-entropy。
            imitation_loss = F.nll_loss(torch.log(action_probs + 1e-8), mb_gt_labels)

            # 总损失：PPO policy + critic + entropy + imitation prior。
            total_loss = (
                policy_loss
                + vf_coef * value_loss
                - ent_coef * entropy
                + alpha * imitation_loss
            )

            optimizer.zero_grad()
            total_loss.backward()

            policy_grad_norm = _parameter_grad_norm(agent.policy_head.parameters())
            value_grad_norm = _parameter_grad_norm(agent.value_head.parameters())
            shared_grad_norm = _parameter_grad_norm(agent.shared.parameters())

            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()
            approx_kl = (mb_old_log_probs - new_log_probs).mean()

            policy_losses.append(float(policy_loss.detach().item()))
            value_losses.append(float(value_loss.detach().item()))
            imitation_losses.append(float(imitation_loss.detach().item()))
            entropies.append(float(entropy.detach().item()))
            total_losses.append(float(total_loss.detach().item()))
            clip_fractions.append(float(clip_fraction.detach().item()))
            approx_kls.append(float(approx_kl.detach().item()))
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
        "policy_grad_norm": float(np.mean(policy_grad_norms)) if policy_grad_norms else 0.0,
        "value_grad_norm": float(np.mean(value_grad_norms)) if value_grad_norms else 0.0,
        "shared_grad_norm": float(np.mean(shared_grad_norms)) if shared_grad_norms else 0.0,
    }


def evaluate_on_validation(
    agent: SelectionAgent,
    codebook: VQCodebook,
    decoder: VQDecoder,
    val_env: TradingEnv,
    device: torch.device,
    max_horizons: int | None = None,
) -> dict[str, Any]:
    """在验证集上评估 SelectionAgent，返回平均 horizon return 和诊断指标。

    # 需求 5.7: 定期在验证集上评估性能

    功能说明:
        在验证阶段，对每个 horizon 取首 bar 状态，使用当前策略贪心选择最优
        archetype（argmax），再通过 frozen decoder 执行整段微动作并累加收益。

        除平均 return 外，还额外输出 gross pnl / execution cost / turnover /
        direct flips / archetype histogram 等诊断项，便于区分“方向错”和“成本过高”。

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

    Returns:
        dict[str, Any]: 平均 return 及执行诊断
    """
    agent.eval()
    num_horizons = val_env.num_horizons

    if num_horizons == 0:
        return {"avg_return": 0.0, "selected_histogram": "[]"}

    if max_horizons is not None:
        num_horizons = min(num_horizons, int(max_horizons))

    selected_archetypes: list[int] = []
    horizon_details: list[dict[str, Any]] = []

    # 使用 tqdm 显示进度条
    for h_idx in tqdm(range(num_horizons), desc="验证集评估"):
        state = val_env.states[h_idx * val_env.horizon]
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_probs, _ = agent(state_t)
            k = torch.argmax(action_probs, dim=-1).item()

        selected_archetypes.append(int(k))

        # 获取选定原型的量化嵌入
        z_q = codebook.embeddings.weight[k].unsqueeze(0)  # (1, code_dim)

        detail = run_horizon_with_decoder(
            val_env, h_idx, decoder, z_q, device, return_details=True
        )
        horizon_details.append(detail)

    metrics = _aggregate_execution_diagnostics(horizon_details)
    metrics["selected_histogram"] = _format_histogram_from_counts(
        _histogram_counts(selected_archetypes, codebook.embeddings.weight.size(0))
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
) -> dict[str, Any]:
    """在训练子集上做 learned / random / oracle / fixed baseline 对照。

    功能说明:
        该诊断不直接参与训练，只用于定位负收益来源：
        - learned selector 是否优于 random；
        - gt oracle 是否明显高于 learned；
        - best fixed archetype 是否已经为负。

    论文相关:
        这一步并不改变论文算法本身，而是对 Section 4.2 的 archetype selector
        做工程诊断，帮助判断瓶颈在 selector 还是在 frozen archetype 基座。
    """
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

    states_np = get_horizon_start_states(train_env, horizon_indices)
    states_t = torch.tensor(states_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        action_probs, _ = agent(states_t)
        learned_actions = torch.argmax(action_probs, dim=-1).detach().cpu().numpy()

    gt_labels = get_ground_truth_labels(
        encoder=encoder,
        codebook=codebook,
        demo_states=demo_states,
        demo_actions=demo_actions,
        demo_rewards=demo_rewards,
        horizon_indices=horizon_indices,
        device=device,
    ).detach().cpu().numpy()

    learned_metrics = _run_policy_on_horizons(
        codebook=codebook,
        decoder=decoder,
        env=train_env,
        horizon_indices=horizon_indices,
        device=device,
        selected_archetypes=learned_actions,
    )

    rng = np.random.default_rng(12345)
    random_actions = rng.integers(0, codebook.embeddings.weight.size(0), size=subset_size, dtype=np.int64)
    random_metrics = _run_policy_on_horizons(
        codebook=codebook,
        decoder=decoder,
        env=train_env,
        horizon_indices=horizon_indices,
        device=device,
        selected_archetypes=random_actions,
    )

    oracle_metrics = _run_policy_on_horizons(
        codebook=codebook,
        decoder=decoder,
        env=train_env,
        horizon_indices=horizon_indices,
        device=device,
        selected_archetypes=gt_labels,
    )

    fixed_returns: list[float] = []
    for archetype_idx in range(codebook.embeddings.weight.size(0)):
        fixed_actions = np.full(subset_size, archetype_idx, dtype=np.int64)
        fixed_metrics = _run_policy_on_horizons(
            codebook=codebook,
            decoder=decoder,
            env=train_env,
            horizon_indices=horizon_indices,
            device=device,
            selected_archetypes=fixed_actions,
        )
        fixed_returns.append(float(fixed_metrics["avg_return"]))

    best_fixed_idx = int(np.argmax(fixed_returns)) if fixed_returns else -1

    return {
        "num_horizons": subset_size,
        "learned_return": float(learned_metrics["avg_return"]),
        "random_return": float(random_metrics["avg_return"]),
        "oracle_return": float(oracle_metrics["avg_return"]),
        "best_fixed_return": float(max(fixed_returns)) if fixed_returns else 0.0,
        "best_fixed_idx": best_fixed_idx,
        "learned_gt_agreement": float(np.mean(learned_actions == gt_labels)) if gt_labels.size > 0 else 0.0,
        "learned_selected_histogram": learned_metrics["selected_histogram"],
        "oracle_label_histogram": _format_histogram_from_counts(
            _histogram_counts(gt_labels, codebook.embeddings.weight.size(0))
        ),
        "fixed_returns": "[" + ", ".join(f"{idx}:{ret:.4f}" for idx, ret in enumerate(fixed_returns)) + "]",
        "learned_avg_gross_pnl": float(learned_metrics["avg_gross_pnl"]),
        "learned_avg_cost": float(learned_metrics["avg_execution_cost"]),
        "learned_avg_turnover": float(learned_metrics["avg_turnover"]),
        "learned_avg_direct_flips": float(learned_metrics["avg_direct_flips"]),
    }


def save_checkpoint(
    save_path: str,
    agent: SelectionAgent,
    optimizer: torch.optim.Optimizer,
    reward_history: list[float],
    best_val_return: float,
    step_count: int,
    config: Any,
    ppo_hparams: dict[str, Any],
) -> None:
    """统一保存 checkpoint。

    功能说明:
        保存当前 SelectionAgent、优化器状态、训练奖励历史、最佳验证表现，
        以及 Phase II 所需的关键超参数，便于恢复训练和对照实验。

    论文相关:
        保存的核心对象仍然围绕论文 Section 4.2：
        高层 selector 参数 + 训练时的 archetype selection 配置。
        这里额外保存 PPO 风格超参数，是为了复现实验时可追溯优化器设定。
    """
    torch.save(
        {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
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
                "phase2_rollout_batch_size": ppo_hparams["rollout_batch_size"],
                "phase2_ppo_epochs": ppo_hparams["ppo_epochs"],
                "phase2_minibatch_size": ppo_hparams["minibatch_size"],
                "phase2_clip_eps": ppo_hparams["clip_eps"],
                "phase2_vf_coef": ppo_hparams["vf_coef"],
                "phase2_ent_coef": ppo_hparams["ent_coef"],
                "phase2_max_grad_norm": ppo_hparams["max_grad_norm"],
                "phase2_diagnostic_horizons": ppo_hparams["diagnostic_horizons"],
            },
        },
        save_path,
    )


def run_training_loop(
    agent: SelectionAgent,
    encoder: VQEncoder,
    codebook: VQCodebook,
    decoder: VQDecoder,
    train_env: TradingEnv,
    val_env: TradingEnv,
    demo_states: np.ndarray,
    demo_actions: np.ndarray,
    demo_rewards: np.ndarray,
    optimizer: torch.optim.Optimizer,
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

    pbar = tqdm(total=total_steps, desc="Phase II 训练", unit="step", dynamic_ncols=True)
    while step_count < total_steps:
        current_batch_size = min(rollout_batch_size, total_steps - step_count)

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
        )
        last_batch_diag = batch["diagnostics"]

        if step_count == 0:
            logger.info(
                "首批 rollout 形状: states=%s, actions=%s, returns=%s, advantages=%s, gt_labels=%s",
                tuple(batch["states"].shape),
                tuple(batch["actions"].shape),
                tuple(batch["returns"].shape),
                tuple(batch["advantages"].shape),
                tuple(batch["gt_labels"].shape),
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

        last_stats = ppo_update(
            agent=agent,
            optimizer=optimizer,
            batch=batch,
            alpha=alpha,
            clip_eps=clip_eps,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            max_grad_norm=max_grad_norm,
            device=device,
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
            "val": f"{best_val_return:.4f}",
        })

        # 日志输出
        if step_count >= next_log_step or step_count == total_steps:
            batch_avg_reward = float(np.mean(batch_returns)) if batch_returns else 0.0
            logger.info(
                "Step %7d/%d — avg_reward=%.4f, batch_reward=%.4f, total=%.4f, policy=%.4f, value=%.4f, imitation=%.4f, entropy=%.4f, clipfrac=%.4f, kl=%.4f, p_gn=%.4f, v_gn=%.4f, shared_gn=%.4f",
                step_count,
                total_steps,
                avg_reward,
                batch_avg_reward,
                last_stats["total_loss"],
                last_stats["policy_loss"],
                last_stats["value_loss"],
                last_stats["imitation_loss"],
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
            val_metrics = evaluate_on_validation(
                agent=agent,
                codebook=codebook,
                decoder=decoder,
                val_env=val_env,
                device=device,
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
                agent=agent,
                encoder=encoder,
                codebook=codebook,
                decoder=decoder,
                train_env=train_env,
                demo_states=demo_states,
                demo_actions=demo_actions,
                demo_rewards=demo_rewards,
                diagnostic_horizons=diagnostic_horizons,
                device=device,
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
    ppo_hparams = get_phase2_hparams(config)

    logger.info("Phase II 训练开始: pair=%s", pair)
    logger.info(
        "超参数: total_steps=%d, lr=%.1e, selection_alpha=%.2f, num_archetypes=%d, discount_factor=%.2f",
        config.phase2_total_steps,
        config.learning_rate,
        config.selection_alpha,
        config.num_archetypes,
        config.discount_factor,
    )
    logger.info(
        "PPO 超参数: rollout_batch=%d, ppo_epochs=%d, minibatch=%d, clip_eps=%.3f, vf_coef=%.3f, ent_coef=%.4f, max_grad_norm=%.2f, diagnostic_horizons=%d",
        ppo_hparams["rollout_batch_size"],
        ppo_hparams["ppo_epochs"],
        ppo_hparams["minibatch_size"],
        ppo_hparams["clip_eps"],
        ppo_hparams["vf_coef"],
        ppo_hparams["ent_coef"],
        ppo_hparams["max_grad_norm"],
        ppo_hparams["diagnostic_horizons"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    # ----------------------------------------------------------------
    # Step 1: 加载 Phase I 模型（编码器 + 码本 + 冻结 Decoder）
    # ----------------------------------------------------------------
    encoder, codebook, decoder = load_phase1_model(config, pair, device)

    # ----------------------------------------------------------------
    # Step 2: 加载特征数据，初始化 TradingEnv
    # ----------------------------------------------------------------
    logger.info("加载特征数据: data_dir=%s, pair=%s", config.data_dir, pair)
    pipeline = FeaturePipeline(config.data_dir, pair)
    train_df, val_df, _ = pipeline.get_state_vector()
    train_prices_df, val_prices_df, _ = pipeline.get_prices()

    train_states = train_df.to_numpy()
    val_states = val_df.to_numpy()
    train_prices = train_prices_df["close"].to_numpy()
    val_prices = val_prices_df["close"].to_numpy()

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
        states_dataframe=train_df,
    )
    val_env = TradingEnv(
        states=val_states,
        prices=val_prices,
        pair=pair,
        horizon=config.horizon,
        states_dataframe=val_df,
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
    # Step 2.5: 加载 DP 示范轨迹（用于 Eq.5 的 ground-truth archetype label）
    # DP 轨迹文件由 Phase I 的 DPPlanner.generate_trajectories() 生成，
    # 前 num_horizons 条与训练环境 horizon 索引 1:1 对齐。
    # ----------------------------------------------------------------
    traj_path = os.path.join(
        config.result_dir, pair, "dp_trajectories", f"{pair}_trajectories.npz"
    )
    if not os.path.exists(traj_path):
        raise FileNotFoundError(
            f"DP 轨迹文件不存在: {traj_path}\n"
            f"请先运行 Phase I 训练: python scripts/train_phase1.py --pair {pair}"
        )

    demo_data = np.load(traj_path)
    demo_states = demo_data["states"]    # (N, h, state_dim)
    demo_actions = demo_data["actions"]  # (N, h)
    demo_rewards = demo_data["rewards"]  # (N, h)
    logger.info(
        "DP 示范轨迹加载完成: %d 条, horizon=%d (训练 env horizons=%d)",
        demo_states.shape[0],
        demo_states.shape[1],
        train_env.num_horizons,
    )

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
    ).to(device)

    logger.info(
        "SelectionAgent 初始化完成: params=%d",
        sum(p.numel() for p in agent.parameters()),
    )

    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

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
    val_interval = max(train_env.num_horizons, train_env.num_horizons*10)  # 每遍历一次训练集或步评估一次
    log_interval = int(ppo_hparams["log_interval"])

    save_dir = os.path.join(config.result_dir, pair, "phase2_archetype_selection")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{pair}_selection_agent.pt")

    best_val_return, reward_history, step_count = run_training_loop(
        agent=agent,
        encoder=encoder,
        codebook=codebook,
        decoder=decoder,
        train_env=train_env,
        val_env=val_env,
        demo_states=demo_states,
        demo_actions=demo_actions,
        demo_rewards=demo_rewards,
        optimizer=optimizer,
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
        reward_history=reward_history,
        best_val_return=best_val_return,
        step_count=step_count,
        config=config,
        ppo_hparams=ppo_hparams,
    )
    logger.info("最终模型已保存到 %s", final_save_path)

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