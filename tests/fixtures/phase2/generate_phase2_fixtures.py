"""生成 Phase II 测试 fixture 数据。

设计文档锚点: Phase II 执行计划 §6 单元测试数据计划。

生成文件:
- market_train.feather (96 行)
- market_val.feather (48 行)
- market_test.feather (48 行)
- market_with_gap.feather (含 timestamp gap)
- market_bad_schema.feather (schema 不匹配)
- market_ood_shift.feather (后半段 feature 分布漂移)

价格场景:
- 前段缓慢上涨。
- 中段震荡横盘。
- 后段下跌并夹杂高波动段。
- 至少包含一个跨 horizon 的持续趋势段。

fixture 约束:
- 需要与 smoke Phase I 的 input_schema.json 保持一致。
- close 仅用于 replay / reward / markout，不进入 selector state。
- close 序列长度必须能覆盖 h+2 行访问。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def generate_market_fixture(
    num_rows: int,
    seed: int = 42,
    scenario: str = "normal",
) -> "polars.DataFrame":
    """生成单个 market fixture DataFrame。"""
    import polars as pl

    rng = np.random.RandomState(seed)

    # 基础价格: 前段上涨、中段横盘、后段下跌
    third = num_rows // 3
    prices = np.zeros(num_rows)
    base = 100.0
    # 上涨段
    for i in range(third):
        base += rng.normal(0.02, 0.1)
        prices[i] = max(base, 50.0)
    # 横盘段
    for i in range(third, 2 * third):
        base += rng.normal(0.0, 0.05)
        prices[i] = max(base, 50.0)
    # 下跌段（含高波动）
    for i in range(2 * third, num_rows):
        base += rng.normal(-0.03, 0.2)
        prices[i] = max(base, 50.0)

    # timestamp: 每分钟一行
    timestamps = list(range(num_rows))
    if scenario == "with_gap":
        # 在中间制造 gap
        gap_start = num_rows // 2
        for i in range(gap_start, num_rows):
            timestamps[i] += 100  # 100 分钟 gap

    # 盘口数据
    spread = 0.1
    data = {
        "timestamp": timestamps,
        "close": prices.tolist(),
    }
    for level in range(1, 6):
        data[f"ask{level}_price"] = (prices + spread * level).tolist()
        data[f"ask{level}_size"] = (rng.uniform(10, 100, num_rows)).tolist()
        data[f"bid{level}_price"] = (prices - spread * level).tolist()
        data[f"bid{level}_size"] = (rng.uniform(10, 100, num_rows)).tolist()

    data["total_trade_volume"] = rng.uniform(1000, 10000, num_rows).tolist()
    data["turnover"] = (prices * rng.uniform(100, 1000, num_rows)).tolist()
    data["open_interest"] = rng.uniform(5000, 50000, num_rows).tolist()

    # 特征列
    data["feature_return_1"] = np.concatenate([[0.0], np.diff(prices) / prices[:-1]]).tolist()
    data["feature_vol_4"] = [
        float(np.std(prices[max(0, i - 4):i + 1])) if i >= 1 else 0.0
        for i in range(num_rows)
    ]
    data["feature_momentum_8"] = [
        float(prices[i] - prices[max(0, i - 8)]) / max(prices[max(0, i - 8)], 1.0)
        for i in range(num_rows)
    ]

    if scenario == "ood_shift":
        # 后半段 feature 分布漂移
        half = num_rows // 2
        data["feature_return_1"] = (
            data["feature_return_1"][:half]
            + (np.array(data["feature_return_1"][half:]) * 5 + 0.1).tolist()
        )

    if scenario == "bad_schema":
        # 缺少 close 列
        del data["close"]

    return pl.DataFrame(data)


def generate_all_fixtures(output_dir: Path) -> None:
    """生成所有 Phase II fixture 文件。"""
    import polars as pl

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 正常 fixtures
    train = generate_market_fixture(96, seed=42, scenario="normal")
    val = generate_market_fixture(48, seed=43, scenario="normal")
    test = generate_market_fixture(48, seed=44, scenario="normal")

    train.write_ipc(output_dir / "market_train.feather")
    val.write_ipc(output_dir / "market_val.feather")
    test.write_ipc(output_dir / "market_test.feather")

    # 特殊 fixtures
    gap = generate_market_fixture(48, seed=45, scenario="with_gap")
    gap.write_ipc(output_dir / "market_with_gap.feather")

    ood = generate_market_fixture(48, seed=46, scenario="ood_shift")
    ood.write_ipc(output_dir / "market_ood_shift.feather")

    print(f"Phase II fixtures 已生成到 {output_dir}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    generate_all_fixtures(output_dir)
