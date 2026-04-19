"""诊断脚本: 自动分析 Phase II archetype 选择质量并输出详细报告

功能:
1. 自动识别有问题的 archetype（表现最差、选择频率异常等）
2. 深入诊断问题原型的参数和行为模式
3. 输出详细的诊断报告（JSON + Markdown）供 AI 分析

排查内容:
- 每个 archetype 在 val 上的平均 horizon return
- selector 的 archetype 选择分布（是否坍缩）
- selector 的概率分布熵（是否过于自信/随机）
- 每个 archetype 对应的 decoder 输出动作分布（long/flat/short 比例）
- 亏损 horizon 的 archetype 分布 vs 盈利 horizon 的分布
- 问题 archetype 被选时的市场特征分析

用法:
    python scripts/diagnose_archetype.py --pair AL --batch-id batch_001 --split val
    python scripts/diagnose_archetype.py --pair AL --batch-id batch_001 --split val --output-dir reports
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, ".")

from src.config import parse_args as parse_config
from src.data.feature_pipeline import FeaturePipeline
from src.env.trading_env import TradingEnv
from src.evaluation.model_loader import load_phase1_model, load_phase2_model
from src.evaluation.inference_runner import generate_base_actions, compute_base_return
from src.utils.progress import should_disable_tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="AL")
    p.add_argument("--batch-id", default="batch_001", dest="batch_id")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--output-dir", default="reports", dest="output_dir",
                   help="诊断报告输出目录（默认 reports）")
    p.add_argument("--cycle-feature-sets", default=None, dest="cycle_feature_sets",
                   help="特征集，逗号分隔，如 short / middle / long（默认跟随 config）")
    p.add_argument("--top-n-problems", type=int, default=3, dest="top_n_problems",
                   help="诊断表现最差的前 N 个 archetype（默认 3）")
    return p.parse_args()


def main():
    args = parse_args()
    config_argv = ["--train-batch-id", args.batch_id]
    if args.cycle_feature_sets:
        config_argv += ["--cycle-feature-sets", args.cycle_feature_sets]
    config = parse_config(config_argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair = args.pair
    split = args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Archetype 自动诊断: {pair} [{split}]  batch={args.batch_id}")
    print(f"{'='*60}\n")

    # 加载模型
    try:
        codebook, decoder, normalizer = load_phase1_model(config, pair, device)
    except RuntimeError as e:
        error_msg = str(e)
        if "size mismatch" in error_msg and "state_dict" in error_msg:
            print(f"\n❌ 模型维度不匹配错误:")
            print(f"{error_msg}\n")
            
            # 尝试从 checkpoint 中提取实际的 state_dim
            import re
            match = re.search(r'torch\.Size\(\[(\d+), (\d+)\]\)', error_msg)
            if match:
                checkpoint_dim = int(match.group(2))
                current_dim = config.state_dim
                print(f"问题分析:")
                print(f"  - Checkpoint 中的模型期望 state_dim = {checkpoint_dim}")
                print(f"  - 当前配置使用 state_dim = {current_dim}")
                print(f"  - 差异: {current_dim - checkpoint_dim} 个特征\n")
                
                print(f"解决方案:")
                print(f"  训练模型时使用的特征集与当前不一致。请使用正确的特征集参数:\n")
                
                # 推测可能的特征集
                if checkpoint_dim == 70:
                    print(f"  python scripts/diagnose_archetype.py --pair {pair} --batch-id {args.batch_id} --split {split} --cycle-feature-sets short")
                elif checkpoint_dim == 73:
                    print(f"  python scripts/diagnose_archetype.py --pair {pair} --batch-id {args.batch_id} --split {split} --cycle-feature-sets middle")
                else:
                    print(f"  python scripts/diagnose_archetype.py --pair {pair} --batch-id {args.batch_id} --split {split} --cycle-feature-sets <正确的特征集>")
                
                print(f"\n  或者检查训练日志确认使用的特征集配置")
            else:
                print(f"错误详情: {error_msg}")
        else:
            print(f"\n❌ 加载 Phase I 模型失败: {e}")
            print(f"\n请检查:")
            print(f"  1. 模型文件是否存在: result/{pair}/{args.batch_id}/phase1_archetype_discovery/{pair}_vq_model.pt")
            print(f"  2. 模型文件是否完整（未损坏）")
            print(f"  3. 是否使用正确的 batch-id")
            import traceback
            traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 加载 Phase I 模型失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    try:
        selection_agent = load_phase2_model(config, pair, device)
    except Exception as e:
        print(f"\n❌ 加载 Phase II 模型失败: {e}")
        print(f"\n请检查:")
        print(f"  1. 模型文件是否存在: result/{pair}/{args.batch_id}/phase2_archetype_selection/{pair}_selection_agent.pt")
        print(f"  2. Phase II 训练是否已完成")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    K = config.num_archetypes

    # 加载数据
    print(f"加载数据: {pair}, 特征集: {config.cycle_features}")
    try:
        pipeline = FeaturePipeline(config.data_dir, pair, cycle_features=config.cycle_features)
        train_df, val_df, test_df = pipeline.get_state_vector()
        _, val_prices_df, test_prices_df = pipeline.get_prices()
    except Exception as e:
        print(f"\n❌ 加载数据失败: {e}")
        print(f"\n请检查:")
        print(f"  1. 数据目录是否存在: {config.data_dir}")
        print(f"  2. 交易对数据是否存在: {pair}")
        print(f"  3. 特征集配置是否正确: {config.cycle_features}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    df = val_df if split == "val" else test_df
    prices_df = val_prices_df if split == "val" else test_prices_df

    states = df.to_numpy()
    prices = prices_df["close"].to_numpy()

    if normalizer is not None:
        states_norm = normalizer.normalize_states(states)
    else:
        states_norm = states

    env = TradingEnv(
        states=states_norm, prices=prices,
        pair=pair, horizon=config.horizon, states_dataframe=df,
        max_positions=config.max_positions,
        commission_rate=config.commission_rate,
    )
    print(f"horizons: {env.num_horizons},  horizon_len: {config.horizon}\n")

    # ---- 逐 horizon 收集诊断数据 ----
    archetype_returns = defaultdict(list)   # k -> [horizon_return, ...]
    archetype_probs_entropy = []            # 每个 horizon 的 selector 熵
    archetype_chosen = []                   # 每个 horizon 选的 k
    archetype_action_dist = defaultdict(lambda: np.zeros(3))  # k -> [short, flat, long] counts

    # 用于市场特征对比: 记录每个 horizon 起始 bar 的原始特征（未归一化）
    # 关注价格动量、波动率代理指标
    horizon_raw_features = []   # list of dict，每个 horizon 一条

    # 加载原始（未归一化）数据用于特征分析
    raw_states = df.to_numpy()  # 原始未归一化

    for h_idx in tqdm(
        range(env.num_horizons),
        desc="诊断中",
        unit="horizon",
        disable=should_disable_tqdm(),
    ):
        h = config.horizon
        start = h_idx * h
        end = min(start + h, len(states_norm))
        horizon_states = states_norm[start:end]

        state_0 = states_norm[start]
        state_0_t = torch.tensor(state_0, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_probs, _ = selection_agent(state_0_t)
            probs_np = action_probs.squeeze(0).cpu().numpy()
            k = int(np.argmax(probs_np))

            # 熵
            entropy = float(-np.sum(probs_np * np.log(probs_np + 1e-8)))

        archetype_chosen.append(k)
        archetype_probs_entropy.append(entropy)

        # 生成 base actions
        e_a_sel_t = codebook.embeddings.weight[k]
        z_q = e_a_sel_t.unsqueeze(0)
        base_actions = generate_base_actions(decoder, z_q, horizon_states, device)

        # horizon return（不含跨 horizon 持仓延续，纯净单 horizon 收益）
        h_return = compute_base_return(env, h_idx, base_actions)
        archetype_returns[k].append(h_return)

        # 动作分布统计
        for a in base_actions:
            archetype_action_dist[k][int(a)] += 1

        # 收集原始市场特征（用 horizon 内所有 bar 计算统计量）
        raw_h = raw_states[start:end]  # (h, feature_dim)
        col_names = df.columns
        close_idx = col_names.index("close") if "close" in col_names else 0
        prices_h = raw_h[:, close_idx]

        # 价格动量: horizon 内价格变化幅度
        price_range_pct = (prices_h.max() - prices_h.min()) / (prices_h.mean() + 1e-8)
        # 方向性: 终价 vs 初价
        price_direction = (prices_h[-1] - prices_h[0]) / (prices_h[0] + 1e-8)
        # 波动率代理: 逐步收益率的标准差
        step_rets = np.diff(prices_h) / (prices_h[:-1] + 1e-8)
        volatility = float(step_rets.std()) if len(step_rets) > 1 else 0.0
        # 趋势强度: |方向| / 波动范围
        trend_strength = abs(price_direction) / (price_range_pct + 1e-8)

        # 成交量相关（如果有）
        vol_idx = col_names.index("total_trade_volume") if "total_trade_volume" in col_names else None
        avg_volume = float(raw_h[:, vol_idx].mean()) if vol_idx is not None else 0.0

        # LOB 不平衡（bid1_size vs ask1_size）
        bid1_idx = col_names.index("bid1_size") if "bid1_size" in col_names else None
        ask1_idx = col_names.index("ask1_size") if "ask1_size" in col_names else None
        if bid1_idx is not None and ask1_idx is not None:
            bid1 = raw_h[:, bid1_idx].mean()
            ask1 = raw_h[:, ask1_idx].mean()
            lob_imbalance = float((bid1 - ask1) / (bid1 + ask1 + 1e-8))
        else:
            lob_imbalance = 0.0

        horizon_raw_features.append({
            "h_idx": h_idx,
            "k": k,
            "h_return": h_return,
            "price_range_pct": float(price_range_pct),
            "price_direction": float(price_direction),
            "volatility": float(volatility),
            "trend_strength": float(trend_strength),
            "avg_volume": avg_volume,
            "lob_imbalance": lob_imbalance,
            "entropy": entropy,
        })

    archetype_chosen = np.array(archetype_chosen)
    archetype_probs_entropy = np.array(archetype_probs_entropy)

    # ---- 自动识别问题 archetype ----
    problem_archetypes = []
    
    # 计算每个 archetype 的统计信息
    archetype_stats = {}
    for k in range(K):
        rets = np.array(archetype_returns[k])
        if len(rets) == 0:
            continue
        
        count = int(np.sum(archetype_chosen == k))
        freq = count / env.num_horizons * 100
        avg_ret = rets.mean()
        win_rate = np.mean(rets > 0) * 100
        
        # 计算问题评分（越高越有问题）
        problem_score = 0
        issues = []
        
        # 1. 负收益
        if avg_ret < 0:
            problem_score += abs(avg_ret) * count  # 加权损失
            issues.append(f"负收益: {avg_ret:.2f}")
        
        # 2. 低胜率
        if win_rate < 40:
            problem_score += (40 - win_rate) * count / 100
            issues.append(f"低胜率: {win_rate:.1f}%")
        
        # 3. 高频率但表现差
        if freq > 10 and avg_ret < 0:
            problem_score += freq * abs(avg_ret)
            issues.append(f"高频负收益: freq={freq:.1f}%")
        
        archetype_stats[k] = {
            "k": k,
            "count": count,
            "freq": freq,
            "avg_return": float(avg_ret),
            "std_return": float(rets.std()),
            "min_return": float(rets.min()),
            "max_return": float(rets.max()),
            "win_rate": win_rate,
            "problem_score": problem_score,
            "issues": issues,
        }
        
        if problem_score > 0:
            problem_archetypes.append((k, problem_score, issues))
    
    # 按问题评分排序
    problem_archetypes.sort(key=lambda x: x[1], reverse=True)
    top_problems = problem_archetypes[:args.top_n_problems]
    
    print(f"\n{'='*60}")
    print(f"🔍 自动识别的问题 archetype (Top {args.top_n_problems}):")
    print(f"{'='*60}")
    for idx, (k, score, issues) in enumerate(top_problems, 1):
        stats = archetype_stats[k]
        print(f"\n{idx}. Archetype k={k} (问题评分: {score:.2f})")
        print(f"   选择频率: {stats['freq']:.1f}% ({stats['count']} 次)")
        print(f"   平均收益: {stats['avg_return']:.2f}")
        print(f"   胜率: {stats['win_rate']:.1f}%")
        print(f"   问题:")
        for issue in issues:
            print(f"     - {issue}")

    # ---- 打印结果 ----

    # 1. Archetype 选择频率
    print("【1】Archetype 选择频率分布")
    print(f"{'k':>4}  {'count':>6}  {'freq%':>7}  {'avg_return':>12}  {'win_rate%':>10}  {'action(S/F/L)':>16}")
    print("-" * 65)
    total_h = env.num_horizons
    for k in range(K):
        count = int(np.sum(archetype_chosen == k))
        freq = count / total_h * 100
        rets = archetype_returns[k]
        avg_ret = np.mean(rets) if rets else 0.0
        win_rate = np.mean(np.array(rets) > 0) * 100 if rets else 0.0
        ad = archetype_action_dist[k]
        total_steps = ad.sum()
        if total_steps > 0:
            s_pct = ad[0] / total_steps * 100
            f_pct = ad[1] / total_steps * 100
            l_pct = ad[2] / total_steps * 100
            action_str = f"{s_pct:.0f}%/{f_pct:.0f}%/{l_pct:.0f}%"
        else:
            action_str = "N/A"
        print(f"{k:>4}  {count:>6}  {freq:>6.1f}%  {avg_ret:>12.2f}  {win_rate:>9.1f}%  {action_str:>16}")

    # 2. Selector 熵统计
    print(f"\n【2】Selector 概率分布熵")
    print(f"  均值: {archetype_probs_entropy.mean():.4f}")
    print(f"  最大熵 (均匀分布 K={K}): {np.log(K):.4f}")
    print(f"  熵/最大熵: {archetype_probs_entropy.mean() / np.log(K) * 100:.1f}%")
    print(f"  熵 < 0.5 的 horizon 比例 (高置信): {np.mean(archetype_probs_entropy < 0.5)*100:.1f}%")
    print(f"  熵 > 2.0 的 horizon 比例 (低置信): {np.mean(archetype_probs_entropy > 2.0)*100:.1f}%")

    # 3. 亏损 horizon 分析
    all_returns = np.array([r for k in range(K) for r in archetype_returns[k]])
    # 重建 per-horizon 数组（按 h_idx 顺序）
    per_horizon_return = np.zeros(total_h)
    idx_counter = defaultdict(int)
    for h_idx in range(total_h):
        k = archetype_chosen[h_idx]
        pos = idx_counter[k]
        per_horizon_return[h_idx] = archetype_returns[k][pos]
        idx_counter[k] += 1

    loss_mask = per_horizon_return < 0
    win_mask = per_horizon_return >= 0
    print(f"\n【3】盈亏 horizon 统计")
    print(f"  总 horizons: {total_h}")
    print(f"  盈利 horizons: {win_mask.sum()} ({win_mask.mean()*100:.1f}%)")
    print(f"  亏损 horizons: {loss_mask.sum()} ({loss_mask.mean()*100:.1f}%)")
    print(f"  平均盈利 return: {per_horizon_return[win_mask].mean():.2f}")
    print(f"  平均亏损 return: {per_horizon_return[loss_mask].mean():.2f}")

    if loss_mask.sum() > 0:
        print(f"\n  亏损 horizon 的 archetype 分布:")
        loss_archetypes = archetype_chosen[loss_mask]
        for k in range(K):
            cnt = int(np.sum(loss_archetypes == k))
            if cnt > 0:
                loss_avg = np.mean(per_horizon_return[loss_mask & (archetype_chosen == k)])
                print(f"    k={k}: {cnt} 次  avg_loss={loss_avg:.2f}")

    # 4. 最差 archetype 详情
    print(f"\n【4】各 archetype 收益分布 (按 avg_return 排序)")
    k_stats = []
    for k in range(K):
        rets = np.array(archetype_returns[k])
        if len(rets) == 0:
            continue
        k_stats.append((k, rets.mean(), rets.std(), rets.min(), rets.max(), len(rets)))
    k_stats.sort(key=lambda x: x[1])
    print(f"{'k':>4}  {'avg':>10}  {'std':>10}  {'min':>10}  {'max':>10}  {'n':>5}")
    print("-" * 55)
    for k, avg, std, mn, mx, n in k_stats:
        print(f"{k:>4}  {avg:>10.2f}  {std:>10.2f}  {mn:>10.2f}  {mx:>10.2f}  {n:>5}")

    # 5. 总结
    total_return_sum = per_horizon_return.sum()
    print(f"\n【5】汇总")
    print(f"  所有 horizon return 之和: {total_return_sum:.2f}")
    print(f"  (注: 不含跨 horizon 持仓延续和手续费累积效应，仅供参考)")

    # 找出拖累最大的 archetype
    worst_k = min(k_stats, key=lambda x: x[1])
    best_k = max(k_stats, key=lambda x: x[1])
    print(f"  最差 archetype: k={worst_k[0]}  avg={worst_k[1]:.2f}  n={worst_k[5]}")
    print(f"  最优 archetype: k={best_k[0]}  avg={best_k[1]:.2f}  n={best_k[5]}")

    dominant_k = int(np.bincount(archetype_chosen).argmax())
    dominant_freq = np.mean(archetype_chosen == dominant_k) * 100
    print(f"  最常选 archetype: k={dominant_k}  freq={dominant_freq:.1f}%")
    if dominant_freq > 50:
        print(f"  ⚠️  selector 严重坍缩到 k={dominant_k}，多样性不足")

    # ---- 【6】对每个问题 archetype 进行详细诊断 ----
    feat_keys = ["price_range_pct", "price_direction", "volatility", "trend_strength",
                 "avg_volume", "lob_imbalance"]
    feat_labels = {
        "price_range_pct":  "价格振幅%",
        "price_direction":  "价格方向",
        "volatility":       "逐步波动率",
        "trend_strength":   "趋势强度",
        "avg_volume":       "平均成交量",
        "lob_imbalance":    "LOB不平衡",
    }
    
    detailed_diagnosis = {
        "metadata": {
            "pair": pair,
            "split": split,
            "batch_id": args.batch_id,
            "timestamp": datetime.now().isoformat(),
            "num_archetypes": K,
            "num_horizons": int(env.num_horizons),
            "horizon_length": config.horizon,
        },
        "summary": {
            "total_return": float(total_return_sum),
            "win_rate": float(win_mask.mean() * 100),
            "avg_entropy": float(archetype_probs_entropy.mean()),
            "max_entropy": float(np.log(K)),
            "dominant_archetype": int(dominant_k),
            "dominant_freq": float(dominant_freq),
            "selector_collapsed": bool(dominant_freq > 50),
        },
        "all_archetypes": {},
        "problem_archetypes": {},
    }
    
    # 保存所有 archetype 的统计信息
    for k, stats in archetype_stats.items():
        ad = archetype_action_dist[k]
        total_steps = ad.sum()
        action_dist = {
            "short_pct": float(ad[0] / total_steps * 100) if total_steps > 0 else 0.0,
            "flat_pct": float(ad[1] / total_steps * 100) if total_steps > 0 else 0.0,
            "long_pct": float(ad[2] / total_steps * 100) if total_steps > 0 else 0.0,
        }
        stats["action_distribution"] = action_dist
        detailed_diagnosis["all_archetypes"][str(k)] = stats
    
    # 对每个问题 archetype 进行详细诊断
    for idx, (focus_k, score, issues) in enumerate(top_problems, 1):
        print(f"\n{'='*80}")
        print(f"【详细诊断 {idx}】Archetype k={focus_k} 市场特征分析")
        print(f"{'='*80}")
        
        focus_rows = [r for r in horizon_raw_features if r["k"] == focus_k]
        other_rows = [r for r in horizon_raw_features if r["k"] != focus_k]
        
        print(f"  k={focus_k} 样本数: {len(focus_rows)}   其他: {len(other_rows)}\n")
        
        # 市场特征对比
        print(f"  {'特征':<20}  {'k='+str(focus_k):>12}  {'其他':>12}  {'差异%':>10}  结论")
        print(f"  {'-'*70}")
        
        feature_comparison = {}
        for fk in feat_keys:
            fv_focus = np.array([r[fk] for r in focus_rows]) if focus_rows else np.array([])
            fv_other = np.array([r[fk] for r in other_rows]) if other_rows else np.array([])
            
            m_focus = fv_focus.mean() if len(fv_focus) > 0 else 0.0
            m_other = fv_other.mean() if len(fv_other) > 0 else 0.0
            diff_pct = (m_focus - m_other) / (abs(m_other) + 1e-10) * 100
            
            # 自动判断
            verdict = ""
            if fk == "trend_strength":
                if m_focus < m_other * 0.8:
                    verdict = "✓ 震荡行情"
                elif m_focus > m_other * 1.2:
                    verdict = "✗ 趋势行情"
                else:
                    verdict = "≈ 相近"
            elif fk == "volatility":
                if m_focus < m_other * 0.8:
                    verdict = "✓ 低波动"
                elif m_focus > m_other * 1.2:
                    verdict = "✗ 高波动"
                else:
                    verdict = "≈ 相近"
            elif fk == "price_range_pct":
                if m_focus < m_other * 0.8:
                    verdict = "✓ 窄幅"
                elif m_focus > m_other * 1.2:
                    verdict = "✗ 宽幅"
                else:
                    verdict = "≈ 相近"
            
            label = feat_labels[fk]
            print(f"  {label:<20}  {m_focus:>12.6f}  {m_other:>12.6f}  {diff_pct:>+9.1f}%  {verdict}")
            
            feature_comparison[fk] = {
                "focus_mean": float(m_focus),
                "other_mean": float(m_other),
                "diff_pct": float(diff_pct),
                "verdict": verdict,
            }
        
        # 盈利 vs 亏损分析
        focus_win = [r for r in focus_rows if r["h_return"] >= 0]
        focus_loss = [r for r in focus_rows if r["h_return"] < 0]
        
        print(f"\n  k={focus_k} 内部: 盈利 {len(focus_win)} 次 vs 亏损 {len(focus_loss)} 次")
        
        win_loss_comparison = {}
        if focus_win and focus_loss:
            print(f"\n  {'特征':<20}  {'盈利均值':>12}  {'亏损均值':>12}  {'差异%':>10}")
            print(f"  {'-'*60}")
            for fk in feat_keys:
                mw = np.mean([r[fk] for r in focus_win])
                ml = np.mean([r[fk] for r in focus_loss])
                diff_pct = (mw - ml) / (abs(ml) + 1e-10) * 100
                print(f"  {feat_labels[fk]:<20}  {mw:>12.6f}  {ml:>12.6f}  {diff_pct:>+9.1f}%")
                
                win_loss_comparison[fk] = {
                    "win_mean": float(mw),
                    "loss_mean": float(ml),
                    "diff_pct": float(diff_pct),
                }
        
        # 价格方向分析
        focus_dirs = np.array([r["price_direction"] for r in focus_rows]) if focus_rows else np.array([])
        
        direction_analysis = {}
        if len(focus_dirs) > 0:
            print(f"\n  k={focus_k} 价格方向分布:")
            up_count = np.sum(focus_dirs > 0.001)
            down_count = np.sum(focus_dirs < -0.001)
            flat_count = np.sum(np.abs(focus_dirs) <= 0.001)
            
            print(f"    上涨: {up_count} 次 ({up_count/len(focus_dirs)*100:.1f}%)")
            print(f"    下跌: {down_count} 次 ({down_count/len(focus_dirs)*100:.1f}%)")
            print(f"    横盘: {flat_count} 次 ({flat_count/len(focus_dirs)*100:.1f}%)")
            
            direction_analysis = {
                "up_count": int(up_count),
                "up_pct": float(up_count/len(focus_dirs)*100),
                "down_count": int(down_count),
                "down_pct": float(down_count/len(focus_dirs)*100),
                "flat_count": int(flat_count),
                "flat_pct": float(flat_count/len(focus_dirs)*100),
            }
            
            # 误判分析
            mismatched = [r for r in focus_rows if abs(r["price_direction"]) > 0.002]
            if mismatched:
                mm_returns = np.array([r["h_return"] for r in mismatched])
                print(f"\n  ⚠️  k={focus_k} 被选但有明显趋势(|dir|>0.2%): {len(mismatched)} 次 "
                      f"({len(mismatched)/len(focus_rows)*100:.1f}%)")
                print(f"     这些 horizon 的平均 return: {mm_returns.mean():.2f}")
                print(f"     亏损比例: {np.mean(mm_returns < 0)*100:.1f}%")
                
                direction_analysis["mismatched"] = {
                    "count": len(mismatched),
                    "pct": float(len(mismatched)/len(focus_rows)*100),
                    "avg_return": float(mm_returns.mean()),
                    "loss_rate": float(np.mean(mm_returns < 0)*100),
                }
        
        # 保存详细诊断结果
        detailed_diagnosis["problem_archetypes"][str(focus_k)] = {
            "rank": idx,
            "problem_score": float(score),
            "issues": issues,
            "stats": archetype_stats[focus_k],
            "feature_comparison": feature_comparison,
            "win_loss_comparison": win_loss_comparison,
            "direction_analysis": direction_analysis,
        }
    
    # ---- 生成报告文件 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON 报告
    json_path = output_dir / f"diagnosis_{pair}_{split}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_diagnosis, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*80}")
    print(f"✅ JSON 报告已保存: {json_path}")
    
    # Markdown 报告
    md_path = output_dir / f"diagnosis_{pair}_{split}_{timestamp}.md"
    generate_markdown_report(md_path, detailed_diagnosis, archetype_stats, top_problems)
    print(f"✅ Markdown 报告已保存: {md_path}")
    print(f"{'='*80}\n")


def generate_markdown_report(path, diagnosis, archetype_stats, top_problems):
    """生成 Markdown 格式的诊断报告"""
    with open(path, "w", encoding="utf-8") as f:
        meta = diagnosis["metadata"]
        summary = diagnosis["summary"]
        
        f.write(f"# Archetype 诊断报告\n\n")
        f.write(f"## 基本信息\n\n")
        f.write(f"- **交易对**: {meta['pair']}\n")
        f.write(f"- **数据集**: {meta['split']}\n")
        f.write(f"- **Batch ID**: {meta['batch_id']}\n")
        f.write(f"- **诊断时间**: {meta['timestamp']}\n")
        f.write(f"- **Archetype 数量**: {meta['num_archetypes']}\n")
        f.write(f"- **Horizon 数量**: {meta['num_horizons']}\n")
        f.write(f"- **Horizon 长度**: {meta['horizon_length']}\n\n")
        
        f.write(f"## 整体表现\n\n")
        f.write(f"- **总收益**: {summary['total_return']:.2f}\n")
        f.write(f"- **胜率**: {summary['win_rate']:.1f}%\n")
        f.write(f"- **Selector 平均熵**: {summary['avg_entropy']:.4f} / {summary['max_entropy']:.4f}\n")
        f.write(f"- **主导 Archetype**: k={summary['dominant_archetype']} (频率: {summary['dominant_freq']:.1f}%)\n")
        if summary['selector_collapsed']:
            f.write(f"- **⚠️ 警告**: Selector 严重坍缩，多样性不足\n")
        f.write(f"\n")
        
        f.write(f"## 所有 Archetype 统计\n\n")
        f.write(f"| k | 频率% | 次数 | 平均收益 | 胜率% | 动作分布(S/F/L) |\n")
        f.write(f"|---|-------|------|----------|-------|----------------|\n")
        
        sorted_archetypes = sorted(archetype_stats.items(), key=lambda x: x[1]['avg_return'])
        for k, stats in sorted_archetypes:
            ad = stats['action_distribution']
            f.write(f"| {k} | {stats['freq']:.1f} | {stats['count']} | "
                   f"{stats['avg_return']:.2f} | {stats['win_rate']:.1f} | "
                   f"{ad['short_pct']:.0f}/{ad['flat_pct']:.0f}/{ad['long_pct']:.0f} |\n")
        f.write(f"\n")
        
        f.write(f"## 问题 Archetype 详细诊断\n\n")
        for idx, (k, score, issues) in enumerate(top_problems, 1):
            prob_data = diagnosis["problem_archetypes"][str(k)]
            stats = prob_data["stats"]
            
            f.write(f"### {idx}. Archetype k={k}\n\n")
            f.write(f"**问题评分**: {score:.2f}\n\n")
            f.write(f"**基本统计**:\n")
            f.write(f"- 选择频率: {stats['freq']:.1f}% ({stats['count']} 次)\n")
            f.write(f"- 平均收益: {stats['avg_return']:.2f}\n")
            f.write(f"- 收益标准差: {stats['std_return']:.2f}\n")
            f.write(f"- 收益范围: [{stats['min_return']:.2f}, {stats['max_return']:.2f}]\n")
            f.write(f"- 胜率: {stats['win_rate']:.1f}%\n\n")
            
            f.write(f"**识别的问题**:\n")
            for issue in issues:
                f.write(f"- {issue}\n")
            f.write(f"\n")
            
            # 市场特征对比
            if "feature_comparison" in prob_data:
                f.write(f"**市场特征对比** (k={k} vs 其他):\n\n")
                f.write(f"| 特征 | k={k} | 其他 | 差异% | 结论 |\n")
                f.write(f"|------|-------|------|-------|------|\n")
                for feat, comp in prob_data["feature_comparison"].items():
                    f.write(f"| {feat} | {comp['focus_mean']:.6f} | {comp['other_mean']:.6f} | "
                           f"{comp['diff_pct']:+.1f} | {comp['verdict']} |\n")
                f.write(f"\n")
            
            # 盈亏对比
            if "win_loss_comparison" in prob_data and prob_data["win_loss_comparison"]:
                f.write(f"**盈利 vs 亏损特征对比**:\n\n")
                f.write(f"| 特征 | 盈利均值 | 亏损均值 | 差异% |\n")
                f.write(f"|------|----------|----------|-------|\n")
                for feat, comp in prob_data["win_loss_comparison"].items():
                    f.write(f"| {feat} | {comp['win_mean']:.6f} | {comp['loss_mean']:.6f} | "
                           f"{comp['diff_pct']:+.1f} |\n")
                f.write(f"\n")
            
            # 方向分析
            if "direction_analysis" in prob_data and prob_data["direction_analysis"]:
                da = prob_data["direction_analysis"]
                f.write(f"**价格方向分布**:\n")
                f.write(f"- 上涨: {da['up_count']} 次 ({da['up_pct']:.1f}%)\n")
                f.write(f"- 下跌: {da['down_count']} 次 ({da['down_pct']:.1f}%)\n")
                f.write(f"- 横盘: {da['flat_count']} 次 ({da['flat_pct']:.1f}%)\n")
                
                if "mismatched" in da:
                    mm = da["mismatched"]
                    f.write(f"\n**⚠️ Selector 误判分析**:\n")
                    f.write(f"- 被选但有明显趋势: {mm['count']} 次 ({mm['pct']:.1f}%)\n")
                    f.write(f"- 这些 horizon 的平均收益: {mm['avg_return']:.2f}\n")
                    f.write(f"- 亏损比例: {mm['loss_rate']:.1f}%\n")
                f.write(f"\n")
        
        f.write(f"## 建议\n\n")
        f.write(f"根据诊断结果，建议关注以下方面:\n\n")
        
        if summary['selector_collapsed']:
            f.write(f"1. **Selector 坍缩**: 考虑增加探索性训练或调整 selector 的损失函数\n")
        
        for idx, (k, score, issues) in enumerate(top_problems[:3], 1):
            prob_data = diagnosis["problem_archetypes"][str(k)]
            f.write(f"{idx+1}. **Archetype k={k}**: ")
            
            # 根据特征分析给出建议
            if "direction_analysis" in prob_data and "mismatched" in prob_data["direction_analysis"]:
                mm_pct = prob_data["direction_analysis"]["mismatched"]["pct"]
                if mm_pct > 30:
                    f.write(f"Selector 误判率高 ({mm_pct:.1f}%)，需要改进特征或 selector 训练\n")
            
            if prob_data["stats"]["avg_return"] < -10:
                f.write(f"严重负收益，考虑重新训练该 archetype 或调整 decoder\n")
            elif prob_data["stats"]["win_rate"] < 30:
                f.write(f"胜率过低，检查动作生成策略\n")
            else:
                f.write(f"需要进一步分析市场特征匹配度\n")
        
        f.write(f"\n---\n")
        f.write(f"*报告生成时间: {meta['timestamp']}*\n")


# ---- 【6】focus_k 市场特征对比分析 ----（旧代码，已被上面的自动诊断替代）


if __name__ == "__main__":
    main()
