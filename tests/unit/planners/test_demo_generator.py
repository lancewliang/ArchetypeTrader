"""``Phase1DemoGenerator`` 单元测试 (含 reject_transition 监控)."""
from __future__ import annotations

import pytest

from src.phase1.config import RejectTransitionHealthConfig
from src.preprocess_data.horizon_builder import HorizonRecord
from src.planners.demo_generator import Phase1DemoGenerator, RejectTransitionExceeded
from src.planners.single_trade_dp import SingleTradeDPPlanner
from src.trading.cost_model import ExecutionBook, LobDepthCostModel
from src.trading.reward_alignment import RewardAlignment


def _book(mark: float, depth: float = 100.0, spread_bps: float = 1.0) -> ExecutionBook:
    factor = spread_bps / 10000.0
    asks = tuple(mark * (1 + factor * (i + 1)) for i in range(5))
    bids = tuple(mark * (1 - factor * (i + 1)) for i in range(5))
    sizes = (depth,) * 5
    return ExecutionBook(
        ask_prices=asks, ask_sizes=sizes,
        bid_prices=bids, bid_sizes=sizes,
        mark_price=mark,
    )


def _make_record(prices, books, sample_id="r0", start_index=0):
    return HorizonRecord(
        sample_id=sample_id,
        start_index=start_index,
        end_index=start_index + len(books) - 1,
        pair="TEST",
        split="train",
        strata_label="up|low|mixed",
        states=[[0.0] for _ in books],
        prices=prices,
        execution_books=books,
    )


def _make_gen(fail_when_exceeded=True):
    cm = LobDepthCostModel(commission_rate=0.0001)
    align = RewardAlignment("paper_formula")
    planner = SingleTradeDPPlanner(cost_model=cm, reward_alignment=align)
    return Phase1DemoGenerator(
        planner=planner,
        health=RejectTransitionHealthConfig(
            max_dataset_reject_rate=0.05,
            max_horizon_reject_rate=0.10,
            fail_when_exceeded=fail_when_exceeded,
        ),
    )


def test_generate_fills_actions_and_rewards():
    gen = _make_gen()
    prices = [100.0 + i for i in range(9)]
    books = [_book(p) for p in prices[:-1]]
    record = _make_record(prices, books)
    horizons, stats = gen.generate([record])
    assert horizons[0].actions is not None
    assert horizons[0].rewards is not None
    assert len(horizons[0].actions) == 8


def test_no_trade_horizon_flag_set_via_actions():
    gen = _make_gen()
    # 横盘 + 高手续费 → DP 全 flat
    cm = LobDepthCostModel(commission_rate=0.05)
    planner = SingleTradeDPPlanner(cost_model=cm, reward_alignment=RewardAlignment("paper_formula"))
    gen2 = Phase1DemoGenerator(planner=planner, health=RejectTransitionHealthConfig(fail_when_exceeded=False))
    prices = [100.0] * 9
    books = [_book(100.0, spread_bps=20.0) for _ in range(8)]
    record = _make_record(prices, books)
    out, _ = gen2.generate([record])
    assert all(a == 1 for a in out[0].actions)


def test_metadata_preserves_strata_and_sample_id():
    gen = _make_gen()
    prices = [100.0 + i * 0.1 for i in range(9)]
    books = [_book(p) for p in prices[:-1]]
    record = _make_record(prices, books, sample_id="abc")
    horizons, _ = gen.generate([record])
    assert horizons[0].sample_id == "abc"
    assert horizons[0].strata_label == "up|low|mixed"


def test_reject_stats_collected_when_depth_thin():
    """五档深度极小 → 大量 reject。"""
    gen = _make_gen(fail_when_exceeded=False)
    prices = [100.0 + i for i in range(9)]
    books = [_book(p, depth=0.1) for p in prices[:-1]]  # 深度不足
    record = _make_record(prices, books)
    _, stats = gen.generate([record])
    # 至少 dataset_reject_rate > 0
    assert stats.dataset_reject_rate > 0.0
    assert stats.reject_by_action_pair


def test_fail_when_dataset_reject_rate_exceeds():
    gen = _make_gen(fail_when_exceeded=True)
    # 极小深度，dataset_reject_rate 必然超阈值
    prices = [100.0 + i for i in range(9)]
    books = [_book(p, depth=0.01) for p in prices[:-1]]
    record = _make_record(prices, books)
    with pytest.raises(RejectTransitionExceeded):
        gen.generate([record])


def test_only_warns_when_fail_when_exceeded_false():
    gen = _make_gen(fail_when_exceeded=False)
    prices = [100.0 + i for i in range(9)]
    books = [_book(p, depth=0.01) for p in prices[:-1]]
    record = _make_record(prices, books)
    horizons, stats = gen.generate([record])
    assert horizons  # 不抛错
    assert stats.dataset_reject_rate > 0.0


def test_parallel_generate_matches_serial_order_and_stats():
    cm = LobDepthCostModel(commission_rate=0.0001)
    planner = SingleTradeDPPlanner(
        cost_model=cm,
        reward_alignment=RewardAlignment("paper_formula"),
    )
    health = RejectTransitionHealthConfig(fail_when_exceeded=False)
    serial = Phase1DemoGenerator(planner=planner, health=health)
    parallel = Phase1DemoGenerator(
        planner=planner,
        health=health,
        max_workers=2,
        worker_chunksize=1,
        parallel_min_horizons=1,
    )

    def records():
        out = []
        for i in range(8):
            prices = [100.0 + i * 0.1 + j * 0.2 for j in range(9)]
            books = [_book(p, depth=100.0) for p in prices[:-1]]
            out.append(_make_record(prices, books, sample_id=f"r{i}", start_index=i))
        return out

    serial_records, serial_stats = serial.generate(records())
    parallel_records, parallel_stats = parallel.generate(records())

    assert [rec.sample_id for rec in parallel_records] == [
        rec.sample_id for rec in serial_records
    ]
    assert [rec.actions for rec in parallel_records] == [
        rec.actions for rec in serial_records
    ]
    assert [rec.rewards for rec in parallel_records] == [
        rec.rewards for rec in serial_records
    ]
    assert parallel_stats.per_horizon_reject_count == serial_stats.per_horizon_reject_count
    assert parallel_stats.per_horizon_reject_rate == serial_stats.per_horizon_reject_rate
    assert parallel_stats.reject_by_action_pair == serial_stats.reject_by_action_pair
