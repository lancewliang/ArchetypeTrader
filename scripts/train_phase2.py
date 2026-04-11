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

import json
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
from src.utils.normalizer import StateNormalizer

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
    """在给定 horizons 上执行指定 archetype 选择结果，并汇总诊断指标。

    性能优化: 使用 batch_decode_actions + vectorized_execute_horizons
    替代逐 horizon 的 Python 循环。
    """
    if len(horizon_indices) != len(selected_archetypes):
        raise ValueError(
            f"horizon_indices 和 selected_archetypes 长度不一致: {len(horizon_indices)} vs {len(selected_archetypes)}"
        )

    archetype_t = torch.tensor(selected_archetypes, dtype=torch.long, device=device)

    all_actions_np = batch_decode_actions(
        decoder=decoder,
        codebook=codebook,
        env=env,
        horizon_indices=horizon_indices,
        archetype_indices=archetype_t,
        device=device,
    )

    _, horizon_details = vectorized_execute_horizons(
        env=env,
        horizon_indices=horizon_indices,
        all_actions=all_actions_np,
        need_diagnostics=True,
    )

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
    vf_coef = float(_cfg(config, "phase2_vf_coef", 0.001))
    ent_coef = float(_cfg(config, "phase2_ent_coef", 0.1))
    max_grad_norm = float(_cfg(config, "phase2_max_grad_norm", 1.0))
    log_interval = int(_cfg(config, "phase2_log_interval", 1000000))
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
    """加载 Phase I 模型（编码器 + 码本 + 冻结 Decoder）+ 归一化统计量。

    Returns:
        encoder, codebook, decoder, normalizer
    """
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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    encoder = VQEncoder(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.lstm_hidden_dim,
        latent_dim=config.latent_dim,
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])

    codebook = VQCodebook(
        num_codes=config.num_archetypes,
        code_dim=config.latent_dim,
    ).to(device)
    codebook.load_state_dict(checkpoint["codebook"])

    decoder = VQDecoder(
        state_dim=config.state_dim,
        code_dim=config.latent_dim,
        hidden_dim=config.lstm_hidden_dim,
        action_dim=config.action_dim,
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder"])

    for param in encoder.parameters():
        param.requires_grad = False
    for param in codebook.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    encoder.eval()
    codebook.eval()
    decoder.eval()

    normalizer = StateNormalizer.from_checkpoint_dict(checkpoint)
    if normalizer is not None:
        logger.info("Phase I 归一化统计量已加载")
    else:
        logger.warning("Phase I checkpoint 中无 norm_stats，跳过归一化")

    logger.info("Phase I 模型加载完成，Encoder、Codebook 和 Decoder 已冻结")
    return encoder, codebook, decoder, normalizer


def run_horizon_with_decoder(
    env: TradingEnv,
    horizon_idx: int,
    decoder: VQDecoder,
    z_q: torch.Tensor,
    device: torch.device,
    return_details: bool = False,
    normalizer: StateNormalizer | None = None,
) -> float | dict[str, Any]:
    """使用冻结 Decoder 在一个 horizon 内执行交易，返回 horizon 总收益。

    # Section 4.2: 冻结 Decoder 生成 micro actions
    # 1. 收集 horizon 内所有状态
    # 2. Decoder 根据状态和 z_q 生成 action logits
    # 3. argmax 得到 micro actions
    # 4. 在 env 中逐步执行，累计 horizon return

    功能说明:
        该函数负责把"高层 archetype 决策"真正落地为一个 horizon 内的
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

    # 归一化 states 后再喂给 decoder
    norm_states = horizon_states
    if normalizer is not None:
        norm_states = normalizer.normalize_states(horizon_states)

    # Decoder 批量生成 action logits
    states_t = torch.tensor(
        norm_states, dtype=torch.float32, device=device
    ).unsqueeze(0)
    # states_t: (1, h, state_dim)

    with torch.no_grad():
        action_logits = decoder(states_t, z_q)  # (1, h, action_dim)
        actions = decoder.decode_with_single_trade_constraint(states_t, z_q).squeeze(0)  # (h,)
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

        commission = float(env.commission_rate * abs(delta_position) * price)
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
def batch_decode_actions(
    decoder: VQDecoder,
    codebook: VQCodebook,
    env: TradingEnv,
    horizon_indices: np.ndarray,
    archetype_indices: torch.Tensor,
    device: torch.device,
    normalizer: StateNormalizer | None = None,
) -> np.ndarray:
    """批量解码: 对一批 horizon 用对应 archetype 生成 micro actions。

    Args:
        normalizer: 若提供，对 env.states 做归一化后再喂给 decoder。

    Returns:
        actions_np: (B, h) 所有 horizon 的 micro action 序列
    """
    h = env.horizon
    B = len(horizon_indices)

    all_states = np.empty((B, h, env.state_dim), dtype=np.float32)
    for i, h_idx in enumerate(horizon_indices):
        start = int(h_idx) * h
        end = min(start + h, len(env.states))
        actual_len = end - start
        all_states[i, :actual_len] = env.states[start:end]
        if actual_len < h:
            all_states[i, actual_len:] = env.states[end - 1]

    if normalizer is not None:
        all_states = normalizer.normalize_states(all_states)

    states_t = torch.tensor(all_states, dtype=torch.float32, device=device)

    z_q_batch = codebook.embeddings.weight[archetype_indices.long()]

    with torch.no_grad():
        actions = decoder.decode_with_single_trade_constraint(states_t, z_q_batch)

    return actions.detach().cpu().numpy()
def vectorized_execute_horizons(
    env: TradingEnv,
    horizon_indices: np.ndarray,
    all_actions: np.ndarray,
    need_diagnostics: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """向量化执行多个 horizon 的 micro actions，返回 horizon returns 和诊断。

    功能说明:
        对 B 个 horizon 同时计算 reward，核心计算（持仓映射、价差、佣金）
        全部用 NumPy 向量化完成，避免逐步 env.step() 的 Python 循环。

        LOB slippage 仍需逐行查 DataFrame（因为 polars row dict 无法批量化），
        但持仓变化/佣金/reward 的主体计算已完全向量化。

    Args:
        env: 交易环境
        horizon_indices: (B,) horizon 索引
        all_actions: (B, h) micro action 序列
        need_diagnostics: 是否计算详细诊断统计（False 时只返回 horizon_return）

    Returns:
        returns: (B,) 每个 horizon 的总收益
        details: 长度 B 的诊断字典列表（need_diagnostics=False 时为仅含 horizon_return 的简化字典）
    """
    B = len(horizon_indices)
    h = env.horizon
    m = env.m

    # 持仓方向映射: action {0,1,2} → direction {-1,0,1}
    DIRECTION_MAP = np.array([-1, 0, 1], dtype=np.int64)  # action 0→-1, 1→0, 2→1

    # (B, h) → 目标持仓序列
    directions = DIRECTION_MAP[all_actions]  # (B, h)
    positions = directions * m  # (B, h) 目标持仓

    # 前一步持仓: 第 0 步的前一步持仓为 0（flat）
    prev_positions = np.zeros_like(positions)
    prev_positions[:, 1:] = positions[:, :-1]

    # 持仓变化量
    delta_positions = positions - prev_positions  # (B, h) 有符号
    abs_delta = np.abs(delta_positions)  # (B, h)

    # 价格序列: 每个 horizon 的 h 步价格和 h 步 next_price
    starts = (horizon_indices * h).astype(np.int64)  # (B,)
    price_indices = starts[:, None] + np.arange(h, dtype=np.int64)[None, :]  # (B, h)
    next_indices = np.minimum(price_indices + 1, len(env.prices) - 1)  # (B, h)

    prices = env.prices[price_indices]  # (B, h)
    next_prices = env.prices[next_indices]  # (B, h)
    price_diff = next_prices - prices  # (B, h)

    # 佣金: δ × |ΔP| × price
    commissions = env.commission_rate * abs_delta * prices  # (B, h)

    # LOB slippage: 需要逐行查 DataFrame（无法完全向量化）
    slippages = np.zeros((B, h), dtype=np.float64)
    if env.states_dataframe is not None:
        # 只对持仓变化的步骤计算 slippage
        change_mask = delta_positions != 0
        for bi in range(B):
            for ti in range(h):
                if change_mask[bi, ti]:
                    global_t = int(price_indices[bi, ti])
                    state_dict = env.states_dataframe.row(global_t, named=True)
                    slippages[bi, ti] = TradingEnv.compute_lob_slippage(
                        int(delta_positions[bi, ti]), state_dict, float(prices[bi, ti])
                    )

    # 总执行损失
    execution_costs = slippages + commissions  # (B, h)

    # Eq. 1: r_step_t = P_t × (p_{t+1} - p_t) - O_t
    rewards = positions * price_diff - execution_costs  # (B, h)

    # horizon returns
    horizon_returns = rewards.sum(axis=1)  # (B,)

    if not need_diagnostics:
        # 快速路径: 只返回 horizon_return，跳过所有诊断统计
        details = [{"horizon_return": float(horizon_returns[i])} for i in range(B)]
        return horizon_returns, details

    # 完整诊断统计
    gross_pnl = (rewards + execution_costs).sum(axis=1)  # (B,)
    exec_cost_total = execution_costs.sum(axis=1)  # (B,)
    commission_total = commissions.sum(axis=1)  # (B,)
    slippage_total = slippages.sum(axis=1)  # (B,)
    turnover_total = abs_delta.sum(axis=1).astype(np.float64)  # (B,)

    # 持仓变化次数
    position_changed = (delta_positions != 0)  # (B, h)
    num_position_changes = position_changed.sum(axis=1)  # (B,)

    # direct flips: old != 0 and new != 0 and sign(old) != sign(new)
    old_nonzero = prev_positions != 0
    new_nonzero = positions != 0
    sign_diff = np.sign(prev_positions) != np.sign(positions)
    num_direct_flips = (old_nonzero & new_nonzero & sign_diff).sum(axis=1)  # (B,)

    # decoder action histogram per horizon
    details: list[dict[str, Any]] = []
    for i in range(B):
        action_hist = [0, 0, 0]
        for a in range(3):
            action_hist[a] = int(np.sum(all_actions[i] == a))
        details.append({
            "horizon_return": float(horizon_returns[i]),
            "gross_pnl": float(gross_pnl[i]),
            "execution_cost_total": float(exec_cost_total[i]),
            "commission_total": float(commission_total[i]),
            "slippage_total": float(slippage_total[i]),
            "num_position_changes": int(num_position_changes[i]),
            "num_direct_flips": int(num_direct_flips[i]),
            "turnover_total": float(turnover_total[i]),
            "num_steps": h,
            "decoder_action_histogram": action_hist,
        })

    return horizon_returns, details


def get_horizon_start_states(
    env: TradingEnv, horizon_indices: np.ndarray,
    normalizer: StateNormalizer | None = None,
) -> np.ndarray:
    """获取一批 horizon 的起始状态（归一化后）。"""
    start_indices = horizon_indices * env.horizon
    states = env.states[start_indices]
    if normalizer is not None:
        states = normalizer.normalize_states(states)
    return states


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

    returns_t = torch.tensor(horizon_returns_np, dtype=torch.float32, device=device)

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

            # Eq.(5): KL(â_sel || π_sel)，advantage-weighted 版本：
            # 只对 advantage > 0 的样本施加 imitation 正则，
            # 避免把 policy 拉向 DP 建议但 env 收益为负的 archetype。
            pos_mask = (mb_advantages > 0).float()  # (mb,)
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

            policy_grad_norm = _parameter_grad_norm(agent.policy_head.parameters())
            shared_grad_norm = _parameter_grad_norm(agent.shared.parameters())
            value_grad_norm = _parameter_grad_norm(agent.value_head.parameters())

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
    states_np = get_horizon_start_states(val_env, horizon_indices)
    states_t = torch.tensor(states_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        action_probs, _ = agent(states_t)
        selected_archetypes = torch.argmax(action_probs, dim=-1).detach().cpu().numpy()

    # 批量 decoder + 向量化执行
    archetype_t = torch.tensor(selected_archetypes, dtype=torch.long, device=device)
    all_actions_np = batch_decode_actions(
        decoder=decoder,
        codebook=codebook,
        env=val_env,
        horizon_indices=horizon_indices,
        archetype_indices=archetype_t,
        device=device,
    )

    _, horizon_details = vectorized_execute_horizons(
        env=val_env,
        horizon_indices=horizon_indices,
        all_actions=all_actions_np,
        need_diagnostics=True,
    )

    metrics = _aggregate_execution_diagnostics(horizon_details)
    metrics["selected_histogram"] = _format_histogram_from_counts(
        _histogram_counts(selected_archetypes, codebook.embeddings.weight.size(0))
    )
    metrics["avg_return"] = metrics.pop("avg_return")

    agent.train()
    return metrics


def _phase2_health_status(metrics: dict[str, Any]) -> tuple[str, str]:
    """给出 Phase II 验证结果的健康度标签与说明。"""
    avg_return = float(metrics.get("avg_return", 0.0))
    avg_cost = float(metrics.get("avg_execution_cost", 0.0))

    if avg_return < 0.0:
        return (
            "bad_negative_return",
            "验证集平均 return 为负，说明第二阶段策略当前方向存在明显问题；直接进入第三阶段通常难以彻底修复。",
        )

    # “微弱收益”判定: 收益仅与执行成本同量级，边际非常薄
    weak_threshold = max(1e-8, abs(avg_cost))
    if avg_return <= weak_threshold:
        return (
            "weak_edge",
            "验证集收益仅与执行成本同量级，属于微弱优势；建议先继续打磨第二阶段再进入第三阶段。",
        )

    return ("healthy", "验证集收益明显高于执行成本，第二阶段模型整体健康。")


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
    critic_optimizer: torch.optim.Optimizer,
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
            "critic_optimizer": critic_optimizer.state_dict(),
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
        if need_diag:
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
            critic_optimizer=critic_optimizer,
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
    logger.info("结果目录批次: %s", config.train_batch_id)

    # ----------------------------------------------------------------
    # Step 1: 加载 Phase I 模型（编码器 + 码本 + 冻结 Decoder）
    # ----------------------------------------------------------------
    encoder, codebook, decoder, normalizer = load_phase1_model(config, pair, device)

    # ----------------------------------------------------------------
    # Step 2: 加载特征数据，初始化 TradingEnv
    # ----------------------------------------------------------------
    logger.info("加载特征数据: data_dir=%s, pair=%s", config.data_dir, pair)
    pipeline = FeaturePipeline(
        config.data_dir, pair, cycle_features=config.cycle_features,
    )
    train_df, val_df, _ = pipeline.get_state_vector()
    train_prices_df, val_prices_df, _ = pipeline.get_prices()

    train_states = train_df.to_numpy()
    val_states = val_df.to_numpy()
    train_prices = train_prices_df["close"].to_numpy()
    val_prices = val_prices_df["close"].to_numpy()

    # 归一化 states（与 Phase 1 训练一致）
    if normalizer is not None:
        train_states = normalizer.normalize_states(train_states)
        val_states = normalizer.normalize_states(val_states)

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
        max_positions=config.max_positions,
        commission_rate=config.train_commission_rate,
    )
    val_env = TradingEnv(
        states=val_states,
        prices=val_prices,
        pair=pair,
        horizon=config.horizon,
        states_dataframe=val_df,
        max_positions=config.max_positions,
        commission_rate=config.train_commission_rate,
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
        config.get_stage_result_dir(pair, "dp_trajectories"), "trajectories.npz",
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

    # 归一化 demo 数据（与 Phase 1 训练时一致）
    if normalizer is not None:
        demo_states = normalizer.normalize_states(demo_states)
        demo_rewards = normalizer.normalize_rewards(demo_rewards)

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
    val_interval = max(train_env.num_horizons, train_env.num_horizons*200)  # 每遍历一次训练集或步评估一次
    log_interval = int(ppo_hparams["log_interval"])

    save_dir = config.get_stage_result_dir(pair, "phase2_archetype_selection")
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

    final_val_metrics = evaluate_on_validation(
        agent=agent,
        codebook=codebook,
        decoder=decoder,
        val_env=val_env,
        device=device,
        max_horizons=eval_max_horizons,
    )
    final_val_return = float(final_val_metrics["avg_return"])

    best_val_metrics: dict[str, Any]
    if os.path.exists(save_path):
        best_ckpt = torch.load(save_path, map_location=device, weights_only=False)
        best_agent = SelectionAgent(
            state_dim=config.state_dim,
            num_archetypes=config.num_archetypes,
        ).to(device)
        best_agent.load_state_dict(best_ckpt["agent"])
        best_agent.eval()
        best_val_metrics = evaluate_on_validation(
            agent=best_agent,
            codebook=codebook,
            decoder=decoder,
            val_env=val_env,
            device=device,
            max_horizons=eval_max_horizons,
        )
    else:
        logger.warning("未找到最优 checkpoint: %s，将使用最终模型验证结果替代", save_path)
        best_val_metrics = dict(final_val_metrics)

    best_status, best_status_msg = _phase2_health_status(best_val_metrics)
    final_status, final_status_msg = _phase2_health_status(final_val_metrics)

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
