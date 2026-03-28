#!/usr/bin/env python
"""Phase I 轨迹审计脚本

用途:
1. 检查 trajectories.npz 中的 exact duplicate 比例。
2. 统计 trade type / entry / exit / holding / return bucket 分布。
3. 在不改论文结构的前提下，先确认训练集 D 是否已经具备足够多样性。

说明:
- 该脚本不修改模型，也不改 validation 阈值。
- 只用于分析 sample n chunks 之后的 demo trajectory 质量。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np

FLAT_ACTION = 1


def _sequence_hash(states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(states).tobytes())
    h.update(np.ascontiguousarray(actions).tobytes())
    h.update(np.ascontiguousarray(rewards).tobytes())
    return h.hexdigest()


def _extract_trade_profile(actions: np.ndarray, rewards: np.ndarray, horizon: int) -> Dict[str, Any]:
    non_flat = np.flatnonzero(actions != FLAT_ACTION)
    total_return = float(np.sum(rewards))
    if non_flat.size == 0:
        return {
            "trade_type": "flat",
            "entry_idx": -1,
            "exit_idx": -1,
            "holding_length": 0,
            "total_return": total_return,
        }

    entry_idx = int(non_flat[0])
    exit_idx = int(non_flat[-1])
    holding_length = int(exit_idx - entry_idx + 1)
    first_trade_action = int(actions[entry_idx])
    trade_type = "short" if first_trade_action == 0 else "long" if first_trade_action == 2 else "flat"
    return {
        "trade_type": trade_type,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "holding_length": holding_length,
        "total_return": total_return,
    }


def _time_bucket(index: int, horizon: int, num_buckets: int = 4) -> str:
    if index < 0:
        return "none"
    clipped = min(max(index, 0), horizon - 1)
    bucket = int((clipped * num_buckets) / max(horizon, 1))
    bucket = min(bucket, num_buckets - 1)
    return f"t{bucket}"


def _holding_bucket(length: int, horizon: int) -> str:
    if length <= 0:
        return "h0"
    ratio = float(length) / float(max(horizon, 1))
    if ratio <= 0.10:
        return "h1"
    if ratio <= 0.25:
        return "h2"
    if ratio <= 0.50:
        return "h3"
    return "h4"


def _quantile_bucket_ids(values: np.ndarray, num_buckets: int = 5) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=np.int32)
    if np.allclose(values, values[0]):
        return np.zeros(values.shape[0], dtype=np.int32)

    quantiles = np.linspace(0.0, 1.0, num_buckets + 1)
    edges = np.quantile(values, quantiles)
    unique_edges = np.unique(np.asarray(edges, dtype=np.float64))
    if unique_edges.size <= 2:
        return np.zeros(values.shape[0], dtype=np.int32)
    return np.digitize(values, unique_edges[1:-1], right=False).astype(np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Phase I demonstration trajectories.") 
    parser.add_argument("--out-json", type=str, default="", help="可选：审计报告输出路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traj_path = "result/ETH/dp_trajectories/trajectories.npz"
    data = np.load(traj_path)
    states = data["states"]
    actions = data["actions"]
    rewards = data["rewards"]
    horizon = int(actions.shape[1]) if actions.ndim == 2 else 0

    hashes: List[str] = []
    trade_counter = Counter()
    bucket_counter = Counter()
    entry_counter = Counter()
    exit_counter = Counter()
    hold_counter = Counter()

    profiles: List[Dict[str, Any]] = []
    returns = rewards.sum(axis=1).astype(np.float64)
    return_bucket_ids = _quantile_bucket_ids(returns, num_buckets=5)

    for idx in range(actions.shape[0]):
        record_hash = _sequence_hash(states[idx], actions[idx], rewards[idx])
        hashes.append(record_hash)

        profile = _extract_trade_profile(actions[idx], rewards[idx], horizon=horizon)
        profile["return_bucket"] = int(return_bucket_ids[idx]) if idx < return_bucket_ids.shape[0] else 0
        profiles.append(profile)

        trade_counter[str(profile["trade_type"])] += 1
        entry_counter[_time_bucket(int(profile["entry_idx"]), horizon=horizon)] += 1
        exit_counter[_time_bucket(int(profile["exit_idx"]), horizon=horizon)] += 1
        hold_counter[_holding_bucket(int(profile["holding_length"]), horizon=horizon)] += 1
        bucket_name = (
            f'{profile["trade_type"]}|{_time_bucket(int(profile["entry_idx"]), horizon=horizon)}|'
            f'{_time_bucket(int(profile["exit_idx"]), horizon=horizon)}|'
            f'{_holding_bucket(int(profile["holding_length"]), horizon=horizon)}|'
            f'r{int(profile["return_bucket"])}'
        )
        bucket_counter[bucket_name] += 1

    unique_hashes = len(set(hashes))
    duplicate_count = len(hashes) - unique_hashes
    report = {
        "num_trajectories": int(actions.shape[0]),
        "horizon": int(horizon),
        "unique_sequence_count": int(unique_hashes),
        "duplicate_count": int(duplicate_count),
        "duplicate_ratio": float(duplicate_count / max(len(hashes), 1)),
        "trade_type_histogram": dict(trade_counter),
        "entry_bucket_histogram": dict(entry_counter),
        "exit_bucket_histogram": dict(exit_counter),
        "holding_bucket_histogram": dict(hold_counter),
        "bucket_histogram_top20": dict(bucket_counter.most_common(20)),
        "return_mean": float(np.mean(returns)) if returns.size > 0 else 0.0,
        "return_std": float(np.std(returns)) if returns.size > 0 else 0.0,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
