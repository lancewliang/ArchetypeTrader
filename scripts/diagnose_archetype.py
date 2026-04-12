"""诊断脚本: 分析 Phase II archetype 选择质量

排查 phase2_eval vs DP 的 gap 来源:
1. 每个 archetype 在 val 上的平均 horizon return
2. selector 的 archetype 选择分布（是否坍缩）
3. selector 的概率分布熵（是否过于自信/随机）
4. 每个 archetype 对应的 decoder 输出动作分布（long/flat/short 比例）
5. 亏损 horizon 的 archetype 分布 vs 盈利 horizon 的分布
6. k=8 被选时的市场特征 vs 其他 archetype（判断是否真的是震荡行情）

用法:
    python scripts/diagnose_archetype.py --pair AL --batch-id batch_001 --split val
    python scripts/diagnose_archetype.py --pair AL --batch-id batch_001 --split val --focus-k 8
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, ".")

from src.config import parse_args as parse_config
from src.data.feature_pipeline import FeaturePipeline
from src.env.trading_env import TradingEnv
from src.evaluation.model_loader import load_phase1_model, load_phase2_model
from src.evaluation.inference_runner import generate_base_actions, compute_base_return


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="AL")
    p.add_argument("--batch-id", default="batch_001", dest="batch_id")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--focus-k", type=int, default=8, dest="focus_k",
                   help="重点分析的 archetype index（默认 8）")
    return p.parse_args()


def main():
    args = parse_args()
    config = parse_config(["--train-batch-id", args.batch_id])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair = args.pair
    split = args.split
    focus_k = args.focus_k

    print(f"\n{'='*60}")
    print(f"Archetype 诊断: {pair} [{split}]  batch={args.batch_id}")
    print(f"{'='*60}\n")

    # 加载模型
    codebook, decoder, normalizer = load_phase1_model(config, pair, device)
    selection_agent = load_phase2_model(config, pair, device)

    K = config.num_archetypes

    # 加载数据
    pipeline = FeaturePipeline(config.data_dir, pair, cycle_features=config.cycle_features)
    train_df, val_df, test_df = pipeline.get_state_vector()
    _, val_prices_df, test_prices_df = pipeline.get_prices()

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

    for h_idx in tqdm(range(env.num_horizons), desc="诊断中", unit="horizon"):
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

    # ---- 【6】focus_k 市场特征对比分析 ----
    feat_keys = ["price_range_pct", "price_direction", "volatility", "trend_strength",
                 "avg_volume", "lob_imbalance"]
    feat_labels = {
        "price_range_pct":  "价格振幅%    (高=波动大)",
        "price_direction":  "价格方向     (正=上涨,负=下跌)",
        "volatility":       "逐步波动率   (高=震荡)",
        "trend_strength":   "趋势强度     (高=趋势,低=震荡)",
        "avg_volume":       "平均成交量",
        "lob_imbalance":    "LOB不平衡    (正=买压,负=卖压)",
    }

    focus_rows = [r for r in horizon_raw_features if r["k"] == focus_k]
    other_rows  = [r for r in horizon_raw_features if r["k"] != focus_k]

    print(f"\n{'='*60}")
    print(f"【6】k={focus_k} 市场特征对比 (k={focus_k} vs 其他所有 archetype)")
    print(f"{'='*60}")
    print(f"  k={focus_k} 样本数: {len(focus_rows)}   其他: {len(other_rows)}\n")

    print(f"  {'特征':<30}  {'k='+str(focus_k)+' 均值':>12}  {'其他均值':>10}  {'差异':>10}  结论")
    print(f"  {'-'*80}")
    for fk in feat_keys:
        fv_focus = np.array([r[fk] for r in focus_rows])
        fv_other = np.array([r[fk] for r in other_rows])
        m_focus = fv_focus.mean()
        m_other = fv_other.mean()
        diff_pct = (m_focus - m_other) / (abs(m_other) + 1e-10) * 100
        if fk == "trend_strength":
            verdict = "✓ 确实震荡" if m_focus < m_other * 0.8 else ("✗ 趋势行情" if m_focus > m_other * 1.2 else "≈ 相近")
        elif fk == "volatility":
            verdict = "✓ 低波动" if m_focus < m_other * 0.8 else ("✗ 高波动" if m_focus > m_other * 1.2 else "≈ 相近")
        elif fk == "price_range_pct":
            verdict = "✓ 窄幅震荡" if m_focus < m_other * 0.8 else ("✗ 宽幅波动" if m_focus > m_other * 1.2 else "≈ 相近")
        else:
            verdict = ""
        label = feat_labels[fk]
        print(f"  {label:<30}  {m_focus:>12.6f}  {m_other:>10.6f}  {diff_pct:>+9.1f}%  {verdict}")

    # k=focus_k 内部：盈利 vs 亏损 horizon 的特征差异
    focus_win  = [r for r in focus_rows if r["h_return"] >= 0]
    focus_loss = [r for r in focus_rows if r["h_return"] < 0]
    print(f"\n  k={focus_k} 内部: 盈利 {len(focus_win)} 次 vs 亏损 {len(focus_loss)} 次")
    if focus_win and focus_loss:
        print(f"\n  {'特征':<30}  {'盈利均值':>10}  {'亏损均值':>10}  差异")
        print(f"  {'-'*65}")
        for fk in feat_keys:
            mw = np.mean([r[fk] for r in focus_win])
            ml = np.mean([r[fk] for r in focus_loss])
            diff_pct = (mw - ml) / (abs(ml) + 1e-10) * 100
            print(f"  {feat_labels[fk]:<30}  {mw:>10.6f}  {ml:>10.6f}  {diff_pct:>+8.1f}%")

    # 价格方向分布对比
    print(f"\n  k={focus_k} 被选时的价格方向分布:")
    focus_dirs = np.array([r["price_direction"] for r in focus_rows])
    print(f"    上涨 horizon (dir>0.001): {np.sum(focus_dirs > 0.001)} 次 "
          f"({np.mean(focus_dirs > 0.001)*100:.1f}%)")
    print(f"    下跌 horizon (dir<-0.001): {np.sum(focus_dirs < -0.001)} 次 "
          f"({np.mean(focus_dirs < -0.001)*100:.1f}%)")
    print(f"    横盘 horizon: {np.sum(np.abs(focus_dirs) <= 0.001)} 次 "
          f"({np.mean(np.abs(focus_dirs) <= 0.001)*100:.1f}%)")

    other_dirs = np.array([r["price_direction"] for r in other_rows])
    print(f"\n  其他 archetype 被选时的价格方向分布:")
    print(f"    上涨 horizon (dir>0.001): {np.sum(other_dirs > 0.001)} 次 "
          f"({np.mean(other_dirs > 0.001)*100:.1f}%)")
    print(f"    下跌 horizon (dir<-0.001): {np.sum(other_dirs < -0.001)} 次 "
          f"({np.mean(other_dirs < -0.001)*100:.1f}%)")
    print(f"    横盘 horizon: {np.sum(np.abs(other_dirs) <= 0.001)} 次 "
          f"({np.mean(np.abs(other_dirs) <= 0.001)*100:.1f}%)")

    # 关键：k=focus_k 被选时有多少 horizon 其实有明显趋势
    mismatched = [r for r in focus_rows if abs(r["price_direction"]) > 0.002]
    print(f"\n  ⚠️  k={focus_k} 被选但价格方向明显(|dir|>0.2%): {len(mismatched)} 次 "
          f"({len(mismatched)/max(len(focus_rows),1)*100:.1f}%)")
    if mismatched:
        mm_returns = np.array([r["h_return"] for r in mismatched])
        print(f"     这些 horizon 的平均 return: {mm_returns.mean():.2f}  "
              f"(亏损比例: {np.mean(mm_returns < 0)*100:.1f}%)")
        print(f"     → 这部分是 selector 误判，把趋势行情当成震荡行情处理")


if __name__ == "__main__":
    main()
