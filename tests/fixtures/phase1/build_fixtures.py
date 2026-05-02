"""为集成测试生成小型市场数据 fixture (确定性).

用 random walk + 周期项让 close 含足够波动；ask/bid 五档围绕 close 展开 ±0.05% × level。
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class FixtureSpec:
    train_rows: int = 2400
    val_rows: int = 720
    test_rows: int = 720
    seed: int = 20260430
    base_price: float = 100.0


def _generate_split(rows: int, base_price: float, seed: int):
    """返回 polars DataFrame；列含 timestamp、close、ask{1..5}_*、bid{1..5}_*、若干 derived。"""
    import polars as pl

    rng = random.Random(seed)
    closes = []
    cur = base_price
    for t in range(rows):
        # 噪声 + 周期项让 DP 能找到合法 single trade
        drift = 0.00005 * math.sin(t / 60.0)
        shock = rng.gauss(0.0, 0.0003)
        cur = cur * (1.0 + drift + shock)
        closes.append(max(cur, 1e-3))

    ask_levels = []
    bid_levels = []
    for level in range(1, 6):
        spread = 0.0005 * level
        ask_levels.append([c * (1.0 + spread) for c in closes])
        bid_levels.append([c * (1.0 - spread) for c in closes])

    sizes = [[100.0 + 10.0 * level for _ in closes] for level in range(1, 6)]

    columns = {
        "timestamp": list(range(rows)),
        "close": closes,
    }
    for i in range(5):
        columns[f"ask{i + 1}_price"] = ask_levels[i]
        columns[f"ask{i + 1}_size"] = sizes[i]
        columns[f"bid{i + 1}_price"] = bid_levels[i]
        columns[f"bid{i + 1}_size"] = sizes[i]
    # derived 因子（保证数值列；schema 校验需要 feature_columns 非空）
    columns["mid_price"] = [(a + b) / 2.0 for a, b in zip(ask_levels[0], bid_levels[0])]
    columns["return_1m"] = [0.0] + [
        (closes[i] - closes[i - 1]) / max(closes[i - 1], 1e-9) for i in range(1, rows)
    ]
    columns["total_trade_volume"] = [1000.0 + (i % 17) for i in range(rows)]
    columns["turnover"] = [columns["total_trade_volume"][i] * closes[i] for i in range(rows)]
    columns["open_interest"] = [5000.0 + (i % 23) for i in range(rows)]
    return pl.DataFrame(columns)


def build_fixtures(
    out_dir: Path, spec: FixtureSpec = FixtureSpec()
) -> Tuple[Path, Path, Path]:
    """生成三份 feather；返回 (train, val, test) 路径。"""
    from src.utils.feather_io import write_ipc

    out_dir.mkdir(parents=True, exist_ok=True)
    train_frame = _generate_split(spec.train_rows, spec.base_price, spec.seed)
    val_frame = _generate_split(spec.val_rows, spec.base_price * 1.05, spec.seed + 1)
    test_frame = _generate_split(spec.test_rows, spec.base_price * 1.10, spec.seed + 2)
    train_path = write_ipc(train_frame, out_dir / "market_train.feather")
    val_path = write_ipc(val_frame, out_dir / "market_val.feather")
    test_path = write_ipc(test_frame, out_dir / "market_test.feather")
    return train_path, val_path, test_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tests/fixtures/phase1")
    args = parser.parse_args()
    out_dir = Path(args.out)
    train, val, test = build_fixtures(out_dir)
    print(f"train: {train}\nval: {val}\ntest: {test}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
