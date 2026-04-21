"""Phase II Rollout 执行模块

本模块提供 Phase II 训练和评估中的 rollout 执行相关功能。

Functions:
    get_horizon_start_states: 获取 horizon 起始状态
    batch_decode_actions: 批量解码 archetype 为 micro actions
    vectorized_execute_horizons: 向量化执行 micro actions 并计算收益
    get_ground_truth_labels: 获取 ground-truth archetype labels

论文相关:
    对应 Section 4.2 的 archetype selection 和 micro action execution。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.env.trading_env import TradingEnv
from src.utils.normalizer import StateNormalizer
from src.phase1.codebook import VQCodebook
from src.phase1.vq_decoder import VQDecoder
from src.phase1.vq_encoder import VQEncoder


def get_horizon_start_states(
    env: TradingEnv, horizon_indices: np.ndarray,
    normalizer: StateNormalizer | None = None,
) -> np.ndarray:
    """获取一批 horizon 的起始状态（归一化后）。

    Args:
        env: 交易环境
        horizon_indices: (B,) horizon 索引数组
        normalizer: 若提供，对状态做归一化

    Returns:
        states: (B, state_dim) 起始状态数组
    """
    start_indices = horizon_indices * env.horizon
    states = env.states[start_indices]
    if normalizer is not None:
        states = normalizer.normalize_states(states)
    return states


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

    # Section 4.2: 冻结 Decoder 生成 micro actions
    # 1. 收集 horizon 内所有状态
    # 2. Decoder 根据状态和 archetype code 生成 action logits
    # 3. 使用 single trade constraint 得到 micro actions

    功能说明:
        该函数是 run_horizon_with_decoder 的批量版本，负责把"高层 archetype 决策"
        转换为低层 micro actions。对一批 horizon，根据选定的 archetype，
        通过冻结的 decoder 生成整段 micro action 序列。

    论文相关:
        - 对应 Section 4.2 中：选定 archetype 后，将其 code e_{a_sel}
          输入 frozen decoder p_theta_d(a_base | s, e_{a_sel})；
        - 批量处理多个 horizon，提高计算效率。

    Args:
        decoder: 冻结的 VQ Decoder
        codebook: 冻结的 VQ Codebook
        env: 交易环境
        horizon_indices: (B,) horizon 索引数组
        archetype_indices: (B,) 每个 horizon 对应的 archetype 索引
        device: 计算设备
        normalizer: 若提供，对 env.states 做归一化后再喂给 decoder

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

    # Section 4.2: 在 env 中执行 micro actions，累计 horizon return
    # 对应 run_horizon_with_decoder 的执行部分，但采用向量化实现

    功能说明:
        该函数是 run_horizon_with_decoder 执行部分的批量版本，负责把 micro actions
        在环境中执行并计算收益。对 B 个 horizon 同时计算 reward，核心计算
        （持仓映射、价差、佣金）全部用 NumPy 向量化完成，避免逐步 env.step() 的
        Python 循环。

        为了排查 reward 为负的来源，额外统计 gross pnl / execution cost /
        turnover / direct flips 等诊断信息。

        LOB slippage 仍需逐行查 DataFrame（因为 polars row dict 无法批量化），
        但持仓变化/佣金/reward 的主体计算已完全向量化。

    论文相关:
        - 对应 Section 4.2 中：在环境中执行 decoder 生成的 micro actions；
        - 函数输出的 horizon_return 对应论文里的 r_t^sel，
          即一个 horizon 内所有 step reward 的求和；
        - 批量处理多个 horizon，大幅提高执行效率。

    Args:
        env: 交易环境
        horizon_indices: (B,) horizon 索引
        all_actions: (B, h) micro action 序列
        need_diagnostics: 是否计算详细诊断统计（False 时只返回 horizon_return）

    Returns:
        returns: (B,) 每个 horizon 的总收益（未折扣）
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

    # LOB slippage: 预提取所有涉及时间步的 LOB 数组，消除逐行 Polars row 查找
    slippages = np.zeros((B, h), dtype=np.float64)
    if env.states_dataframe is not None:
        change_mask = delta_positions != 0

        # 收集所有需要计算 slippage 的全局时间步索引（去重）
        # price_indices shape (B, h)，flatten 后取唯一值，避免重复提取
        all_global_t = price_indices[change_mask]  # 只取有持仓变化的位置
        if all_global_t.size > 0:
            unique_ts = np.unique(all_global_t)

            # 一次性从 DataFrame 提取所有需要的行的 LOB 列为 NumPy 数组
            # 用 polars 的列式 select + to_numpy，比逐行 .row() 快一个数量级
            lob_df = env.states_dataframe[unique_ts.tolist()]
            ap = lob_df.select(TradingEnv.LOB_ASK_PRICE_COLS).to_numpy()  # (U, 5)
            as_ = lob_df.select(TradingEnv.LOB_ASK_SIZE_COLS).to_numpy()  # (U, 5)
            bp = lob_df.select(TradingEnv.LOB_BID_PRICE_COLS).to_numpy()  # (U, 5)
            bs = lob_df.select(TradingEnv.LOB_BID_SIZE_COLS).to_numpy()   # (U, 5)

            # 建立 global_t → 数组行索引的映射，O(1) 查找
            t_to_row = {int(t): i for i, t in enumerate(unique_ts)}

            for bi in range(B):
                for ti in range(h):
                    if not change_mask[bi, ti]:
                        continue
                    global_t = int(price_indices[bi, ti])
                    row = t_to_row[global_t]
                    dp = int(delta_positions[bi, ti])
                    mark = float(prices[bi, ti])
                    abs_dp = abs(dp)

                    # 5-level LOB walk（纯 NumPy 数组索引，无 dict 查找）
                    lvl_p = ap[row] if dp > 0 else bp[row]
                    lvl_s = as_[row] if dp > 0 else bs[row]

                    qty = float(abs_dp)
                    fill_cash = 0.0
                    last_p = mark
                    for lv in range(5):
                        lp, ls = float(lvl_p[lv]), float(lvl_s[lv])
                        if lp <= 0 or ls <= 0:
                            continue
                        last_p = lp
                        fill_qty = min(qty, ls)
                        fill_cash += fill_qty * lp
                        qty -= fill_qty
                        if qty <= 0:
                            break
                    if qty > 0:
                        fill_cash += qty * last_p

                    slip = (fill_cash - abs_dp * mark) if dp > 0 else (abs_dp * mark - fill_cash)
                    slippages[bi, ti] = max(slip, 0.0)

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

    Args:
        encoder: 冻结的 VQ Encoder
        codebook: 冻结的 VQ Codebook
        demo_states: 示范轨迹状态数组
        demo_actions: 示范轨迹动作数组
        demo_rewards: 示范轨迹奖励数组
        horizon_indices: (B,) horizon 索引数组
        device: 计算设备

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
