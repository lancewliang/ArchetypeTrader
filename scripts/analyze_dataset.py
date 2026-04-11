"""
验证集 & 测试集分析脚本
- 数据分布统计（均值/标准差/分位数）
- 分布偏移检测（KS 检验 + PSI）
- Baseline 最大可能收益（DP Oracle）
- 结果写入 docs/dataset_analysis_{pair}.txt
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
from scipy import stats

from src.config import Config
from src.data.feature_pipeline import FeaturePipeline, FIXED_FEATURES


# ─────────────────────────────────────────────
# PSI 计算
# ─────────────────────────────────────────────
def compute_psi(base: np.ndarray, compare: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index：衡量两个分布的偏移程度。
    PSI < 0.1: 稳定；0.1~0.2: 轻微偏移；> 0.2: 显著偏移。
    """
    base = base[np.isfinite(base)]
    compare = compare[np.isfinite(compare)]
    if len(base) == 0 or len(compare) == 0:
        return float("nan")

    breakpoints = np.percentile(base, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return float("nan")

    base_counts = np.histogram(base, bins=breakpoints)[0]
    cmp_counts = np.histogram(compare, bins=breakpoints)[0]

    base_pct = base_counts / base_counts.sum()
    cmp_pct = cmp_counts / cmp_counts.sum()

    eps = 1e-6
    base_pct = np.where(base_pct == 0, eps, base_pct)
    cmp_pct = np.where(cmp_pct == 0, eps, cmp_pct)

    psi = np.sum((cmp_pct - base_pct) * np.log(cmp_pct / base_pct))
    return float(psi)


# ─────────────────────────────────────────────
# 分布统计
# ─────────────────────────────────────────────
def describe_series(arr: np.ndarray) -> dict:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {}
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "skew": float(stats.skew(arr)),
        "kurt": float(stats.kurtosis(arr)),
        "nan_pct": float(np.isnan(arr).mean() * 100),
    }


def fmt_desc(d: dict) -> str:
    return (
        f"  count={d['count']:>7}  mean={d['mean']:>12.4f}  std={d['std']:>12.4f}\n"
        f"  min={d['min']:>12.4f}  p25={d['p25']:>12.4f}  p50={d['p50']:>12.4f}\n"
        f"  p75={d['p75']:>12.4f}  p95={d['p95']:>12.4f}  max={d['max']:>12.4f}\n"
        f"  skew={d['skew']:>8.4f}  kurt={d['kurt']:>8.4f}  nan%={d['nan_pct']:>6.2f}%"
    )


# ─────────────────────────────────────────────
# DP Oracle 最大收益
# ─────────────────────────────────────────────
def dp_oracle_return(
    prices: np.ndarray, h: int, max_pos: int, commission_rate: float
) -> np.ndarray:
    """对每个 horizon 枚举所有单次换仓策略，返回每段最优收益。"""
    n = len(prices) // h
    results = []
    for i in range(n):
        seg = prices[i * h : (i + 1) * h]
        best = 0.0
        for pos_a in [-max_pos, 0, max_pos]:
            for pos_b in [-max_pos, 0, max_pos]:
                if pos_a == pos_b:
                    continue
                for t in range(h + 1):
                    r = 0.0
                    prev = 0
                    for step in range(h):
                        cur = pos_a if step < t else pos_b
                        cost = commission_rate * abs(cur - prev) * seg[step]
                        price_diff = seg[step + 1] - seg[step] if step + 1 < h else 0.0
                        r += cur * price_diff - cost
                        prev = cur
                    best = max(best, r)
        results.append(best)
    return np.array(results)


# ─────────────────────────────────────────────
# 主分析函数
# ─────────────────────────────────────────────
def analyze(pair: str, config: Config, out_dir: str = "docs") -> None:
    print(f"\n{'='*60}")
    print(f"  分析品种: {pair}")
    print(f"{'='*60}")

    pipeline = FeaturePipeline(
        config.data_dir, pair, cycle_features=config.cycle_features
    )

    pipeline._load_data()
    raw_val = pipeline._raw_val
    raw_test = pipeline._raw_test
    raw_train = pipeline._raw_train

    val_prices = raw_val["close"].to_numpy()
    test_prices = raw_test["close"].to_numpy()
    train_prices = raw_train["close"].to_numpy()

    feature_cols = [
        c
        for c in raw_val.columns
        if raw_val[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        and c not in ("timestamp",)
    ]

    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"数据集分析报告 — {pair}")
    lines.append(f"生成时间: {ts}")
    lines.append("=" * 70)

    # ── 1. 基本信息
    lines.append("\n[1] 数据集基本信息")
    lines.append("-" * 50)
    for name, df in [("train", raw_train), ("val", raw_val), ("test", raw_test)]:
        dt_col = df["datetime"]
        lines.append(
            f"  {name:5s}: {df.shape[0]:>7} 行 × {df.shape[1]} 列  "
            f"时间范围: {dt_col[0]} ~ {dt_col[-1]}"
        )

    # ── 2. 价格分布
    lines.append("\n[2] close 价格分布")
    lines.append("-" * 50)
    for name, arr in [("train", train_prices), ("val", val_prices), ("test", test_prices)]:
        d = describe_series(arr)
        lines.append(f"\n  [{name}]")
        lines.append(fmt_desc(d))

    # ── 3. 特征分布偏移（val vs test）
    lines.append("\n[3] 特征分布偏移检测（val 为基准 → test）")
    lines.append("-" * 50)
    lines.append(
        f"  {'特征':<45} {'KS统计量':>10} {'KS p值':>10} {'PSI':>8} {'偏移判断':>10}"
    )
    lines.append("  " + "-" * 88)

    shift_summary = []
    for col in feature_cols:
        v = raw_val[col].to_numpy().astype(float)
        t = raw_test[col].to_numpy().astype(float)
        v = v[np.isfinite(v)]
        t = t[np.isfinite(t)]
        if len(v) < 10 or len(t) < 10:
            continue
        ks_stat, ks_p = stats.ks_2samp(v, t)
        psi = compute_psi(v, t)
        if psi > 0.2:
            flag = "显著偏移"
        elif psi > 0.1:
            flag = "轻微偏移"
        else:
            flag = "稳定"
        lines.append(
            f"  {col:<45} {ks_stat:>10.4f} {ks_p:>10.4e} {psi:>8.4f} {flag:>10}"
        )
        shift_summary.append((col, ks_stat, ks_p, psi, flag))

    top_shift = sorted(shift_summary, key=lambda x: x[3], reverse=True)[:10]
    lines.append("\n  [PSI 最高 Top 10 特征]")
    lines.append(f"  {'特征':<45} {'PSI':>8} {'偏移判断':>10}")
    for col, _, _, psi, flag in top_shift:
        lines.append(f"  {col:<45} {psi:>8.4f} {flag:>10}")

    # ── 4. 关键特征 val vs test 均值/标准差对比
    lines.append("\n[4] 关键特征统计对比（val vs test）")
    lines.append("-" * 50)
    key_features = FIXED_FEATURES + config.cycle_features
    key_features = [c for c in key_features if c in feature_cols]
    lines.append(
        f"  {'特征':<40} {'val_mean':>12} {'test_mean':>12} "
        f"{'val_std':>10} {'test_std':>10} {'均值偏差%':>10}"
    )
    lines.append("  " + "-" * 98)
    for col in key_features:
        v = raw_val[col].to_numpy().astype(float)
        t = raw_test[col].to_numpy().astype(float)
        v = v[np.isfinite(v)]
        t = t[np.isfinite(t)]
        if len(v) == 0 or len(t) == 0:
            continue
        vm, tm = np.mean(v), np.mean(t)
        vs, ts_ = np.std(v), np.std(t)
        bias = (tm - vm) / (abs(vm) + 1e-8) * 100
        lines.append(
            f"  {col:<40} {vm:>12.4f} {tm:>12.4f} {vs:>10.4f} {ts_:>10.4f} {bias:>9.2f}%"
        )

    # ── 5. DP Oracle 最大收益 Baseline
    lines.append("\n[5] DP Oracle 最大可能收益 Baseline")
    lines.append("-" * 50)
    h = config.horizon
    max_pos = config.max_positions.get(pair, 1)
    cr = config.train_commission_rate

    lines.append(f"  horizon={h}, max_position={max_pos}, commission_rate={cr}")
    lines.append("")

    print("  计算 val DP oracle...")
    val_oracle = dp_oracle_return(val_prices, h, max_pos, cr)
    print("  计算 test DP oracle...")
    test_oracle = dp_oracle_return(test_prices, h, max_pos, cr)

    for name, oracle in [("val", val_oracle), ("test", test_oracle)]:
        d = describe_series(oracle)
        lines.append(f"  [{name} DP Oracle per-horizon 收益]")
        lines.append(fmt_desc(d))
        lines.append(f"  总收益 sum={oracle.sum():.2f}  horizons数={len(oracle)}")
        lines.append("")

    lines.append("  [val vs test Oracle 对比]")
    lines.append(
        f"  val  : mean={val_oracle.mean():.2f}, median={np.median(val_oracle):.2f}, "
        f"sum={val_oracle.sum():.0f}"
    )
    lines.append(
        f"  test : mean={test_oracle.mean():.2f}, median={np.median(test_oracle):.2f}, "
        f"sum={test_oracle.sum():.0f}"
    )
    ratio = test_oracle.mean() / (val_oracle.mean() + 1e-8)
    lines.append(f"  test/val mean 比值: {ratio:.3f}")

    # ── 6. 简单 Baseline 策略
    lines.append("\n[6] 简单 Baseline 策略（无需训练）")
    lines.append("-" * 50)

    def eval_strategy(prices, h, max_pos, cr, strategy="long"):
        n = len(prices) // h
        returns = []
        for i in range(n):
            seg = prices[i * h : (i + 1) * h]
            pos = max_pos if strategy == "long" else -max_pos
            r = 0.0
            prev = 0
            for step in range(h):
                cost = cr * abs(pos - prev) * seg[step]
                price_diff = seg[step + 1] - seg[step] if step + 1 < h else 0.0
                r += pos * price_diff - cost
                prev = pos
            returns.append(r)
        return np.array(returns)

    for name, prices in [("val", val_prices), ("test", test_prices)]:
        long_r = eval_strategy(prices, h, max_pos, cr, "long")
        short_r = eval_strategy(prices, h, max_pos, cr, "short")
        lines.append(f"  [{name}]")
        lines.append(
            f"  Buy & Hold (long) : mean={long_r.mean():>8.2f}, "
            f"sum={long_r.sum():>10.0f}, win_rate={np.mean(long_r>0)*100:.1f}%"
        )
        lines.append(
            f"  Sell & Hold(short): mean={short_r.mean():>8.2f}, "
            f"sum={short_r.sum():>10.0f}, win_rate={np.mean(short_r>0)*100:.1f}%"
        )
        lines.append("")

    # ── 7. 偏移总结
    n_stable = sum(1 for _, _, _, psi, _ in shift_summary if psi <= 0.1)
    n_mild = sum(1 for _, _, _, psi, _ in shift_summary if 0.1 < psi <= 0.2)
    n_severe = sum(1 for _, _, _, psi, _ in shift_summary if psi > 0.2)
    lines.append("\n[7] 偏移总结")
    lines.append("-" * 50)
    lines.append(f"  总特征数: {len(shift_summary)}")
    lines.append(f"  稳定 (PSI<=0.1)    : {n_stable} ({n_stable/max(len(shift_summary),1)*100:.1f}%)")
    lines.append(f"  轻微偏移 (0.1~0.2) : {n_mild} ({n_mild/max(len(shift_summary),1)*100:.1f}%)")
    lines.append(f"  显著偏移 (PSI>0.2) : {n_severe} ({n_severe/max(len(shift_summary),1)*100:.1f}%)")

    # ── 写入文件
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"dataset_analysis_{pair}.txt")
    report = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  报告已写入: {out_path}")
    print(report)


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="验证集 & 测试集分布分析")
    parser.add_argument("--pair", type=str, default="AL", help="交易品种，默认 AL")
    parser.add_argument("--out_dir", type=str, default="docs", help="报告输出目录")
    args = parser.parse_args()

    config = Config()
    analyze(args.pair, config, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
