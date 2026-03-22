#!/usr/bin/env python
"""评估脚本 — 三阶段完整推理与指标计算

# 需求: 8.7, 8.8, 8.9
#
# 流程:
# 1. 加载 Phase I 模型（码本 + 冻结 Decoder）
# 2. 加载 Phase II 模型（冻结 SelectionAgent）
# 3. 加载 Phase III 模型（RefinementAgent）
# 4. 在测试集（2024-01-01 至 2024-09-01）上运行完整推理流程
# 5. 按交易对分别输出评估结果
# 6. 保存结果到 result/evaluation/
#
# 用法:
#   python scripts/evaluate.py
#   python scripts/evaluate.py --pair BTC
"""

import json
import os
import sys

import numpy as np
import torch

from src.config import parse_args
from src.data.feature_pipeline import FeaturePipeline
from src.env.trading_env import TradingEnv
from src.evaluation.metrics import EvaluationEngine
from src.phase1.codebook import VQCodebook
from src.phase1.vq_decoder import VQDecoder
from src.phase2.selection_agent import SelectionAgent
from src.phase3.policy_adapter import PolicyAdapter
from src.phase3.refinement_agent import RefinementAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_phase1_model(config, pair: str, device: torch.device):
    """加载 Phase I 模型（码本 + 冻结 Decoder）。

    Returns:
        codebook: VQCodebook（冻结，eval 模式）
        decoder: VQDecoder（冻结，eval 模式）
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

    for param in codebook.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    codebook.eval()
    decoder.eval()
    return codebook, decoder


def load_phase2_model(config, pair: str, device: torch.device):
    """加载 Phase II 模型（冻结 SelectionAgent）。

    Returns:
        selection_agent: SelectionAgent（冻结，eval 模式）
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

    for param in selection_agent.parameters():
        param.requires_grad = False
    selection_agent.eval()
    return selection_agent


def load_phase3_model(config, pair: str, device: torch.device):
    """加载 Phase III 模型（RefinementAgent）。

    Returns:
        refinement_agent: RefinementAgent（eval 模式）
    """
    beta1 = config.refinement_beta1
    model_path = os.path.join(
        config.result_dir,
        "phase3_archetype_refinement",
        f"{pair}_refinement_agent_beta{beta1}.pt",
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Phase III 模型文件不存在: {model_path}\n"
            f"请先运行 Phase III 训练: python scripts/train_phase3.py --pair {pair}"
        )

    logger.info("加载 Phase III 模型: %s", model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    context_dim = config.latent_dim + 3  # code_dim + a_base + R_arche + τ_remain
    refinement_agent = RefinementAgent(
        market_dim=config.state_dim,
        context_dim=context_dim,
    ).to(device)
    refinement_agent.load_state_dict(checkpoint["agent"])

    for param in refinement_agent.parameters():
        param.requires_grad = False
    refinement_agent.eval()
    return refinement_agent


def generate_base_actions(
    decoder: VQDecoder,
    z_q: torch.Tensor,
    horizon_states: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """使用冻结 Decoder 生成 horizon 内的 base actions。

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
    R_arche: float,
    device: torch.device,
    horizon: int,
):
    """在一个 horizon 内执行完整三阶段推理（无梯度）。

    # Section 4.3: Step-level policy adapter inference
    # 每个 step: refinement agent 观测 state + context → 输出 a_ref
    #            → policy adapter 计算 final action → env step

    Returns:
        step_returns: 每步收益列表
    """
    state = env.reset(horizon_idx)

    h = len(base_actions)
    step_returns = []
    a_base_prev = int(base_actions[0])
    has_adjusted = False

    for step_idx in range(h):
        a_base = int(base_actions[step_idx])

        # 构建 s_ref1 和 s_ref2
        s_ref1 = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)

        tau_remain = (h - step_idx) / h
        context = np.concatenate([
            e_a_sel,
            np.array([a_base], dtype=np.float32),
            np.array([R_arche], dtype=np.float32),
            np.array([tau_remain], dtype=np.float32),
        ])
        s_ref2 = torch.tensor(
            context, dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            action_probs, _ = refinement_agent(s_ref1, s_ref2)
            # 推理时使用 argmax（贪心策略）
            a_ref_idx = torch.argmax(action_probs, dim=-1).item()
            a_ref = a_ref_idx - 1  # 0→-1, 1→0, 2→1

        a_final, has_adjusted = policy_adapter.compute_final_action(a_base, a_base_prev, a_ref, has_adjusted)

        next_state, reward, done, _ = env.step(a_final)
        step_returns.append(reward)

        state = next_state
        a_base_prev = a_base

        if done:
            break

    return step_returns


def evaluate_pair(
    config,
    pair: str,
    device: torch.device,
) -> dict:
    """对单个交易对执行完整评估。

    # 需求 8.7: 在测试集上运行完整三阶段推理
    # 需求 8.8: 按交易对分别输出评估结果

    Args:
        config: 全局配置
        pair: 交易对名称
        device: 计算设备

    Returns:
        评估结果字典
    """
    logger.info("=" * 50)
    logger.info("评估交易对: %s", pair)
    logger.info("=" * 50)

    # 加载三阶段模型
    codebook, decoder = load_phase1_model(config, pair, device)
    selection_agent = load_phase2_model(config, pair, device)
    refinement_agent = load_phase3_model(config, pair, device)

    # 加载特征数据
    logger.info("加载特征数据: data_dir=%s, pair=%s", config.data_dir, pair)
    pipeline = FeaturePipeline(config.data_dir, pair, config)
    states = pipeline.get_state_vector()  # (T, 45)

    # 使用第一列作为价格代理
    # [NOTE: 论文未明确指定价格列，使用 states 第 0 列作为价格]
    prices = states[:, 0].copy()

    # 按时间划分，使用测试集
    _, _, test_states = pipeline.split_by_date(states)
    test_prices = prices[len(states) - len(test_states):]

    logger.info(
        "测试集: states shape=%s, prices shape=%s",
        test_states.shape,
        test_prices.shape,
    )

    # 创建测试环境
    test_env = TradingEnv(
        states=test_states,
        prices=test_prices,
        pair=pair,
        horizon=config.horizon,
    )
    logger.info("TradingEnv 初始化完成: test_horizons=%d", test_env.num_horizons)

    if test_env.num_horizons == 0:
        logger.warning("交易对 %s 测试集 horizon 数量为 0，跳过", pair)
        return {"pair": pair, "error": "no test horizons"}

    # 在每个 horizon 上运行完整推理
    all_step_returns = []

    for h_idx in range(test_env.num_horizons):
        h = test_env.horizon
        start = h_idx * h
        end = min(start + h, len(test_env.states))
        horizon_states = test_env.states[start:end]

        # Phase II: Selection agent 选择原型
        state_0 = test_env.states[start]
        state_0_t = torch.tensor(
            state_0, dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            action_probs, _ = selection_agent(state_0_t)
            k = torch.argmax(action_probs, dim=-1).item()

        # 获取选定原型的量化嵌入
        e_a_sel_t = codebook.embeddings.weight[k]
        z_q = e_a_sel_t.unsqueeze(0)
        e_a_sel = e_a_sel_t.detach().cpu().numpy()

        # Phase I (frozen decoder): 生成 base actions
        base_actions = generate_base_actions(
            decoder, z_q, horizon_states, device
        )

        # 计算 R_base（用于上下文）
        R_arche = compute_base_return(test_env, h_idx, base_actions)

        # Phase III: Refinement agent 精炼
        step_returns = run_horizon_inference(
            env=test_env,
            horizon_idx=h_idx,
            base_actions=base_actions,
            refinement_agent=refinement_agent,
            policy_adapter=PolicyAdapter(),
            e_a_sel=e_a_sel,
            R_arche=R_arche,
            device=device,
            horizon=h,
        )

        all_step_returns.extend(step_returns)

    # 使用 EvaluationEngine 计算所有指标
    returns_array = np.array(all_step_returns, dtype=np.float64)
    engine = EvaluationEngine(annualization_factor=config.annualization_factor)
    metrics = engine.evaluate(returns_array)

    # 添加元信息
    result = {
        "pair": pair,
        "test_start": config.test_start,
        "test_end": config.test_end,
        "num_horizons": test_env.num_horizons,
        "num_steps": len(all_step_returns),
        "beta1": config.refinement_beta1,
        **metrics,
    }

    # 打印结果
    logger.info("评估结果 [%s]:", pair)
    logger.info("  Total Return (TR):          %.6f", metrics["total_return"])
    logger.info("  Annual Volatility (AVOL):   %.6f", metrics["annual_volatility"])
    logger.info("  Max Drawdown (MDD):         %.6f", metrics["max_drawdown"])
    logger.info("  Annual Sharpe Ratio (ASR):  %.6f", metrics["annual_sharpe_ratio"])
    logger.info("  Annual Calmar Ratio (ACR):  %.6f", metrics["annual_calmar_ratio"])
    logger.info("  Annual Sortino Ratio (ASoR):%.6f", metrics["annual_sortino_ratio"])

    return result


def main() -> None:
    # 解析配置
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("评估开始，使用设备: %s", device)

    # 需求 8.9: 保存结果到 result/evaluation/
    save_dir = os.path.join(config.result_dir, "evaluation")
    os.makedirs(save_dir, exist_ok=True)

    all_results = {}

    for pair in config.pairs:
        try:
            result = evaluate_pair(config, pair, device)
            all_results[pair] = result

            # 保存单个交易对结果
            pair_path = os.path.join(save_dir, f"{pair}_results.json")
            with open(pair_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info("结果已保存: %s", pair_path)

        except FileNotFoundError as e:
            logger.error("交易对 %s 评估失败: %s", pair, e)
            all_results[pair] = {"pair": pair, "error": str(e)}
        except Exception as e:
            logger.error("交易对 %s 评估异常: %s", pair, e)
            all_results[pair] = {"pair": pair, "error": str(e)}

    # 保存汇总结果
    summary_path = os.path.join(save_dir, "all_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("汇总结果已保存: %s", summary_path)

    # 打印汇总表格
    logger.info("=" * 70)
    logger.info("评估汇总")
    logger.info("=" * 70)
    logger.info(
        "%-6s %10s %10s %10s %10s %10s %10s",
        "Pair", "TR", "AVOL", "MDD", "ASR", "ACR", "ASoR",
    )
    logger.info("-" * 70)
    for pair, res in all_results.items():
        if "error" in res:
            logger.info("%-6s  ERROR: %s", pair, res["error"][:50])
        else:
            logger.info(
                "%-6s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f",
                pair,
                res["total_return"],
                res["annual_volatility"],
                res["max_drawdown"],
                res["annual_sharpe_ratio"],
                res["annual_calmar_ratio"],
                res["annual_sortino_ratio"],
            )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
