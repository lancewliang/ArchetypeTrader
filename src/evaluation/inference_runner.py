"""InferenceRunner — 三阶段推理执行

从 evaluate.py 中抽取 generate_base_actions、compute_base_return、
run_horizon_inference、evaluate_pair 等核心推理逻辑。
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.config import Config, parse_args
from src.data.feature_pipeline import FeaturePipeline
from src.env.trading_env import TradingEnv
from src.evaluation.metrics import EvaluationEngine
from src.evaluation.model_loader import load_phase1_model, load_phase2_model, load_phase3_model
from src.evaluation.bt_verifier import BacktraderVerifier
from src.evaluation.portfolio_tracker import PortfolioTracker
from src.evaluation.trade_auditor import TradeAuditor
from src.phase1.vq_decoder import VQDecoder
from src.phase3.policy_adapter import PolicyAdapter
from src.phase3.refinement_agent import RefinementAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_base_actions(
    decoder: VQDecoder,
    z_q: torch.Tensor,
    horizon_states: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """使用冻结 Decoder 生成 horizon 内的 base actions。

    Returns:
        base_actions: (h,) 值域 {0, 1, 2}
    """
    states_t = torch.tensor(
        horizon_states, dtype=torch.float32, device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        actions = decoder.decode_with_single_trade_constraint(states_t, z_q).squeeze(0)

    return actions.cpu().numpy()


def compute_base_return(
    env: TradingEnv, horizon_idx: int, base_actions: np.ndarray,
) -> float:
    """使用 base actions 在 env 中执行，返回 horizon 总收益 R_base。"""
    env.reset(horizon_idx)
    total_return = 0.0
    for step_idx in range(len(base_actions)):
        action = int(base_actions[step_idx])
        _, reward, done, _ = env.step(action)
        total_return += reward
        if done:
            break
    return total_return


def run_horizon_inference(
    env: TradingEnv,
    horizon_idx: int,
    base_actions: np.ndarray,
    refinement_agent: RefinementAgent,
    policy_adapter: PolicyAdapter,
    e_a_sel: np.ndarray,
    device: torch.device,
    horizon: int,
    tracker: PortfolioTracker,
) -> List[float]:
    """在一个 horizon 内执行完整三阶段推理。

    Args:
        tracker: PortfolioTracker 实例，管理跨 horizon 的资金与持仓

    Returns:
        step_rate_returns: 每步收益率列表
    """
    state = env.reset(horizon_idx)
    t_start = horizon_idx * env.horizon
    initial_price = env.prices[t_start]

    # R_arche 归一化分母: m × p_0（初始名义价值），与训练一致
    notional = float(env.m) * float(initial_price)
    if notional <= 0.0:
        notional = 1.0

    # 跨 horizon 智能平仓: 同方向延续时跳过，方向改变时收取手续费+滑点
    first_action = int(base_actions[0])
    prev_position = tracker.state.position
    # 计算平仓滑点（平仓 delta: 多头卖出为负，空头买回为正）
    settle_slippage = 0.0
    if prev_position != 0 and env.states_dataframe is not None:
        settle_delta = -prev_position  # 平仓方向
        state_dict = env.states_dataframe.row(t_start, named=True)
        settle_slippage = round(
            TradingEnv.compute_lob_slippage(settle_delta, state_dict, initial_price), 2,
        )

    # settle 前记录 portfolio_value，以便将 settle 成本计入收益率
    pre_settle_value = tracker.compute_total_value(
        prev_position, initial_price,
    )[2]
    if pre_settle_value <= 0.0:
        pre_settle_value = 1.0

    tracker.settle_previous_horizon(
        initial_price, t_start,
        new_first_action=first_action,
        m=env.m,
        commission_rate=env.commission_rate,
        slippage=settle_slippage,
    )

    # 从 tracker 获取实际持仓（可能因同方向延续而非 0）
    current_position = tracker.state.position
    # 同步 env 内部持仓状态
    env._position = current_position

    h = len(base_actions)
    step_rate_returns: List[float] = []
    a_base_prev = int(base_actions[0])
    has_adjusted = False
    cumulative_reward = 0.0  # R_arche: 逐步累积实时收益（与训练一致）

    # portfolio value 用于收益率计算: 使用 tracker 的实际总资产
    portfolio_value = tracker.compute_total_value(
        current_position, initial_price,
    )[2]  # total_value
    if portfolio_value <= 0.0:
        portfolio_value = 1.0

    # 将 settle 产生的成本（手续费+滑点）计入收益率序列
    if abs(portfolio_value - pre_settle_value) > 1e-6:
        settle_return = (portfolio_value - pre_settle_value) / pre_settle_value
        step_rate_returns.append(settle_return)

    for step_idx in range(h):
        a_base = int(base_actions[step_idx])

        # 构建 refinement agent 输入
        s_ref1 = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        tau_remain = (h - step_idx) / h
        normalized_reward = cumulative_reward / notional
        context = np.concatenate([
            e_a_sel,
            np.array([a_base], dtype=np.float32),
            np.array([normalized_reward], dtype=np.float32),
            np.array([tau_remain], dtype=np.float32),
        ])
        s_ref2 = torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_probs, _ = refinement_agent(s_ref1, s_ref2)
            a_ref_idx = torch.argmax(action_probs, dim=-1).item()
            a_ref = a_ref_idx - 1

        a_final, has_adjusted = policy_adapter.compute_final_action(
            a_base, a_base_prev, a_ref, has_adjusted,
        )

        old_position = current_position
        t = t_start + step_idx
        exec_price = float(env.prices[t])

        next_state, reward, done, info = env.step(a_final)
        cumulative_reward += reward  # 更新 R_arche: 逐步累积实时收益
        new_position = info["position"]
        delta_position = new_position - old_position

        # 手续费和滑点
        if delta_position == 0:
            commission, slippage = 0.0, 0.0
        else:
            abs_delta = abs(delta_position)
            commission = round(env.commission_rate * abs_delta * exec_price, 2)
            if env.states_dataframe is not None:
                state_dict = env.states_dataframe.row(t, named=True)
                # 换仓 (多→空 或 空→多) 拆成平仓+开仓两笔，各自独立计算滑点
                if old_position != 0 and new_position != 0 and (
                    (old_position > 0) != (new_position > 0)
                ):
                    close_delta = -old_position  # 平仓方向
                    open_delta = new_position    # 开仓方向
                    slippage_close = TradingEnv.compute_lob_slippage(
                        close_delta, state_dict, exec_price,
                    )
                    slippage_open = TradingEnv.compute_lob_slippage(
                        open_delta, state_dict, exec_price,
                    )
                    slippage = round(slippage_close + slippage_open, 2)
                else:
                    slippage = round(TradingEnv.compute_lob_slippage(
                        delta_position, state_dict, exec_price,
                    ), 2)
            else:
                slippage = 0.0

        # 更新资金
        tracker.update_cash_for_trade(old_position, new_position, exec_price, t)

        # 记录操作
        tracker.record_step(t, a_final, exec_price, old_position, new_position, commission, slippage)

        current_position = new_position

        # 收益率: 基于 tracker 实际总资产变动
        _, _, new_total_value, _ = tracker.compute_total_value(
            new_position, exec_price,
        )
        rate_return = (new_total_value - portfolio_value) / portfolio_value
        step_rate_returns.append(rate_return)
        portfolio_value = new_total_value
        if portfolio_value <= 0.0:
            portfolio_value = 1e-8

        state = next_state
        a_base_prev = a_base

        if done:
            break

    return step_rate_returns


def evaluate_pair(
    config: Config | None = None,
    pair: str = "ETH",
    device: torch.device | None = None,
) -> dict:
    """对单个交易对执行完整评估。

    Returns:
        评估结果字典
    """
    if config is None:
        config = parse_args([])
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 50)
    logger.info("评估交易对: %s", pair)
    logger.info("=" * 50)

    # 加载三阶段模型
    codebook, decoder, normalizer = load_phase1_model(config, pair, device)
    selection_agent = load_phase2_model(config, pair, device)
    refinement_agent = load_phase3_model(config, pair, device)

    # 加载特征数据
    logger.info("加载特征数据: data_dir=%s, pair=%s", config.data_dir, pair)
    pipeline = FeaturePipeline(
        config.data_dir, pair, cycle_features=config.cycle_features,
    )
    _, _, test_df = pipeline.get_state_vector()
    _, _, test_prices_df = pipeline.get_prices()

    test_states = test_df.to_numpy()
    test_prices = test_prices_df["close"].to_numpy()

    # 归一化 states（与 Phase 1 训练一致）
    if normalizer is not None:
        test_states = normalizer.normalize_states(test_states)

    logger.info(
        "测试集: states shape=%s, prices shape=%s",
        test_states.shape, test_prices.shape,
    )

    # 创建测试环境
    test_env = TradingEnv(
        states=test_states, prices=test_prices,
        pair=pair, horizon=config.horizon, states_dataframe=test_df,
        max_positions=config.max_positions,
        commission_rate=config.commission_rate,
    )
    logger.info("TradingEnv 初始化完成: test_horizons=%d", test_env.num_horizons)

    if test_env.num_horizons == 0:
        logger.warning("交易对 %s 测试集 horizon 数量为 0，跳过", pair)
        return {"pair": pair, "error": "no test horizons"}

    # 初始资金
    initial_capital = float(test_env.m) * float(test_prices[0])
    tracker = PortfolioTracker(initial_capital)
    logger.info("初始资金: %.2f (m=%d × price=%.6f)", initial_capital, test_env.m, test_prices[0])

    all_step_returns: List[float] = []

    for h_idx in tqdm(range(test_env.num_horizons), desc=f"评估 {pair}", unit="horizon"):
        h = test_env.horizon
        start = h_idx * h
        end = min(start + h, len(test_env.states))
        horizon_states = test_env.states[start:end]

        # Phase II: 选择原型
        state_0 = test_env.states[start]
        state_0_t = torch.tensor(state_0, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_probs, _ = selection_agent(state_0_t)
            k = torch.argmax(action_probs, dim=-1).item()

        e_a_sel_t = codebook.embeddings.weight[k]
        z_q = e_a_sel_t.unsqueeze(0)
        e_a_sel = e_a_sel_t.detach().cpu().numpy()

        # Phase I: 生成 base actions
        base_actions = generate_base_actions(decoder, z_q, horizon_states, device)

        # Phase III: Refinement
        step_returns = run_horizon_inference(
            env=test_env,
            horizon_idx=h_idx,
            base_actions=base_actions,
            refinement_agent=refinement_agent,
            policy_adapter=PolicyAdapter(),
            e_a_sel=e_a_sel,
            device=device,
            horizon=h,
            tracker=tracker,
        )
        all_step_returns.extend(step_returns)

    # 最终平仓: 评估结束时强制平掉所有仓位，收取手续费+滑点
    final_pos = tracker.state.position
    if final_pos != 0:
        last_t = test_env.num_horizons * test_env.horizon - 1
        last_valid_idx = min(last_t, len(test_prices) - 1)
        final_price = float(test_prices[last_valid_idx])
        settle_delta = -final_pos
        settle_slippage = 0.0
        if test_env.states_dataframe is not None:
            state_dict = test_env.states_dataframe.row(last_valid_idx, named=True)
            settle_slippage = round(
                TradingEnv.compute_lob_slippage(settle_delta, state_dict, final_price), 2,
            )
        settle_idx = last_valid_idx + 1

        # 记录平仓前的 portfolio_value
        pre_final_value = tracker.compute_total_value(final_pos, final_price)[2]
        if pre_final_value <= 0.0:
            pre_final_value = 1.0

        tracker.settle_previous_horizon(
            final_price, settle_idx,
            new_first_action=1,  # flat
            m=test_env.m,
            commission_rate=test_env.commission_rate,
            slippage=settle_slippage,
        )
        logger.info("最终平仓: pos=%d → 0 @ %.6f, 手续费+滑点已扣除", final_pos, final_price)

        # 将最终平仓成本计入收益率
        post_final_value = tracker.compute_total_value(0, final_price)[2]
        if abs(post_final_value - pre_final_value) > 1e-6:
            final_settle_return = (post_final_value - pre_final_value) / pre_final_value
            all_step_returns.append(final_settle_return)

    # 计算指标
    returns_array = np.array(all_step_returns, dtype=np.float64)
    engine = EvaluationEngine(annualization_factor=config.annualization_factor)
    metrics = engine.evaluate(returns_array)

    result = {
        "pair": pair,
        "test_start": config.test_start,
        "test_end": config.test_end,
        "num_horizons": test_env.num_horizons,
        "num_steps": len(all_step_returns),
        "beta1": config.refinement_beta1,
        **metrics,
    }

    # 导出 CSV（每 50000 行一个文件，数字编号）
    csv_save_dir = config.get_stage_result_dir(pair, "evaluation")
    os.makedirs(csv_save_dir, exist_ok=True)
    csv_fields = [
        "state_index", "action", "action_label", "execution_price",
        "trade_quantity", "position_after", "avg_hold_price",
        "commission", "slippage", "position_change_pnl",
        "cash", "holding_value", "short_debt", "total_value", "profit", "side",
    ]
    chunk_size = 50000
    total_records = len(tracker.records)
    num_chunks = (total_records + chunk_size - 1) // chunk_size or 1
    for chunk_idx in range(num_chunks):
        start_row = chunk_idx * chunk_size
        end_row = min(start_row + chunk_size, total_records)
        csv_path = os.path.join(csv_save_dir, f"{pair}_operations_{chunk_idx + 1}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(tracker.records[start_row:end_row])
        logger.info("CSV 已保存: %s (%d 条)", csv_path, end_row - start_row)
    logger.info("共 %d 条记录，分 %d 个文件", total_records, num_chunks)

    # 交易审计: 详细统计 + 一致性检查
    auditor = TradeAuditor(tracker.records, initial_capital)
    audit_report = auditor.audit()
    result["trade_audit"] = audit_report

    # Backtrader 交叉验证: 逐项对比 bt 与 tracker/CSV 的分项数据
    # 多加 2 个 bar: 1 个用于 set_coc 偏移对比，1 个用于最终平仓
    bt_prices = np.append(test_prices, [test_prices[-1]] * 2)
    bt_verifier = BacktraderVerifier(
        records=tracker.records,
        prices=bt_prices,
        initial_capital=initial_capital,
        m=test_env.m,
        commission_rate=test_env.commission_rate,
        tolerance=1.0,
    )
    bt_report = bt_verifier.run(tracker_stats=audit_report["statistics"])
    result["bt_verification"] = bt_report

    # 打印结果
    logger.info("评估结果 [%s]:", pair)
    logger.info("  Total Return (TR):          %.6f", metrics["total_return"])
    logger.info("  Annual Volatility (AVOL):   %.6f", metrics["annual_volatility"])
    logger.info("  Max Drawdown (MDD):         %.6f", metrics["max_drawdown"])
    logger.info("  Annual Sharpe Ratio (ASR):  %.6f", metrics["annual_sharpe_ratio"])
    logger.info("  Annual Calmar Ratio (ACR):  %.6f", metrics["annual_calmar_ratio"])
    logger.info("  Annual Sortino Ratio (ASoR):%.6f", metrics["annual_sortino_ratio"])

    return result
