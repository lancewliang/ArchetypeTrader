"""``Phase1ReplayEvaluator`` 单元测试."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.preprocess_data.horizon_builder import HorizonRecord
from src.phase1.evaluation.replay import Phase1ReplayEvaluator
from src.trading.cost_model import ExecutionBook, LobDepthCostModel
from src.trading.env import TradingEnv
from src.trading.reward_alignment import RewardAlignment


def _book(mark, depth=100.0, spread_bps=2.0):
    factor = spread_bps / 10000.0
    return ExecutionBook(
        ask_prices=tuple(mark * (1 + factor * (i + 1)) for i in range(5)),
        ask_sizes=(depth,) * 5,
        bid_prices=tuple(mark * (1 - factor * (i + 1)) for i in range(5)),
        bid_sizes=(depth,) * 5,
        mark_price=mark,
    )


def _record(sample_id="r0", h=4):
    prices = [100.0 + i * 0.2 for i in range(h + 1)]
    books = [_book(p) for p in prices[:h]]
    rec = HorizonRecord(
        sample_id=sample_id,
        start_index=0,
        end_index=h - 1,
        pair="TEST",
        split="val",
        strata_label="up|low|mixed",
        states=[[0.1, 0.2] for _ in range(h)],
        prices=prices,
        execution_books=books,
        actions=[1, 2, 2, 2],
        rewards=[0.0, 0.0, 0.0, 0.0],
    )
    return rec


def _factory():
    cm = LobDepthCostModel(commission_rate=0.0001)
    align = RewardAlignment("paper_formula")
    return TradingEnv(cost_model=cm, reward_alignment=align, max_position=1)


def test_teacher_replay_matches_action_count():
    evaluator = Phase1ReplayEvaluator(env_factory=_factory)
    rec = _record()
    record = evaluator.replay_dp_teacher(rec)
    assert len(record.teacher_step_returns) == 4
    assert record.teacher_actions == [1, 2, 2, 2]


def test_teacher_replay_raises_when_no_actions():
    rec = _record()
    rec.actions = None
    evaluator = Phase1ReplayEvaluator(env_factory=_factory)
    with pytest.raises(RuntimeError):
        evaluator.replay_dp_teacher(rec)


def test_boundary_replay_empty_when_single_horizon():
    evaluator = Phase1ReplayEvaluator(env_factory=_factory)
    res = evaluator.evaluate_horizon_boundaries([_record()], [[1, 2, 2, 2]])
    assert res.horizon_boundary_position_consistency == 1.0


def test_boundary_replay_records_cost_when_misaligned():
    evaluator = Phase1ReplayEvaluator(env_factory=_factory)
    rec_a = _record("a")
    rec_b = _record("b")
    res = evaluator.evaluate_horizon_boundaries([rec_a, rec_b], [[1, 2, 2, 2], [1, 0, 0, 0]])
    # a 末仓 long → b 首步 flat 后再到 short → 必然有换仓成本
    assert res.horizon_boundary_turnover_cost > 0


def test_online_sequence_replay_inherits_terminal_position():
    class FixedDecoder(torch.nn.Module):
        def forward(self, states, z_q):
            logits = torch.zeros(states.shape[0], states.shape[1], 3, device=states.device)
            logits[:, :, 2] = 1.0
            return logits

    evaluator = Phase1ReplayEvaluator(env_factory=_factory)
    rec_a = _record("a")
    rec_b = _record("b")
    records = evaluator.replay_student_online_sequence(
        [rec_a, rec_b],
        FixedDecoder(),
        torch.ones(1, 2),
        [0, 0],
    )
    assert len(records) == 2
    assert records[0].student_actions == [2, 2, 2, 2]
    # 第二段首步从上一段 long 仓继承而来，不应重新从 flat 建仓。
    assert records[1].cost_paid < records[0].cost_paid


def test_online_sequence_replay_uses_streaming_decoder_path():
    class StreamingOnlyDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state_proj = torch.nn.Linear(2, 2)
            self.lstm = torch.nn.LSTM(input_size=4, hidden_size=2, batch_first=True)
            self.head = torch.nn.Linear(2, 3)
            with torch.no_grad():
                self.state_proj.weight.zero_()
                self.state_proj.bias.zero_()
                for param in self.lstm.parameters():
                    param.zero_()
                self.head.weight.zero_()
                self.head.bias[:] = torch.tensor([0.0, 0.0, 1.0])

        def forward(self, states, z_q):
            raise AssertionError("online sequence replay must not decode a full horizon")

    evaluator = Phase1ReplayEvaluator(env_factory=_factory)
    records = evaluator.replay_student_online_sequence(
        [_record("streaming")],
        StreamingOnlyDecoder(),
        torch.ones(1, 2),
        [0],
    )

    assert records[0].student_actions == [2, 2, 2, 2]
