"""诊断脚本: 分析 Phase II archetype 选择质量

排查 phase2_eval vs DP 的 gap 来源:
1. 每个 archetype 在 val 上的平均 horizon return
2. selector 的 archetype 选择分布（是否坍缩）
3. selector 的概率分布熵（是否过于自信/随机）
4. 每个 archetype 对应的 decoder 输出动作分布（long/flat/short 比例）
5. 亏损 horizon 的 archetype 分布 vs 盈利 horizon 的分布

用法:
    python scripts/diagnose_archetype.py --pair AL --batch_id batch_001 --split val
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
    return p.parse_args()


def main():
    args = parse_args()
    config = parse_config(["--train-batch-id", args.batch_id])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair = args.pair
    split = args.split

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


if __name__ == "__main__":
    main()
