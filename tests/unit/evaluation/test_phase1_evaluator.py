"""``Phase1Evaluator`` 单元测试."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.config.phase1_config import (
    CodebookConfig,
    CodebookHealthConfig,
    EncoderInputConfig,
    ModelConfig,
)
from src.data.dataset import Phase1DemoDataset
from src.data.horizon_builder import HorizonRecord
from src.evaluation.phase1_evaluator import Phase1Evaluator
from src.evaluation.phase1_replay import Phase1ReplayEvaluator
from src.models.vq_archetype import VQArchetypeModel
from src.trading.cost_model import ExecutionBook, LobDepthCostModel
from src.trading.env import TradingEnv
from src.trading.reward_alignment import RewardAlignment


def _book(mark, spread_bps=2.0):
    factor = spread_bps / 10000.0
    return ExecutionBook(
        ask_prices=tuple(mark * (1 + factor * (i + 1)) for i in range(5)),
        ask_sizes=(100.0,) * 5,
        bid_prices=tuple(mark * (1 - factor * (i + 1)) for i in range(5)),
        bid_sizes=(100.0,) * 5,
        mark_price=mark,
    )


def _record(sample_id, h=4):
    prices = [100.0 + i * 0.2 for i in range(h + 1)]
    return HorizonRecord(
        sample_id=sample_id,
        start_index=0,
        end_index=h - 1,
        pair="TEST",
        split="val",
        strata_label="up|low|mixed",
        states=[[0.1, 0.2] for _ in range(h)],
        prices=prices,
        execution_books=[_book(p) for p in prices[:h]],
        actions=[1, 2, 2, 2],
        rewards=[0.0, 0.0, 0.0, 0.0],
    )


def _make_model():
    enc = EncoderInputConfig(state_adapter_dim=8, action_embedding_dim=4, reward_embedding_dim=4, fusion_dim=16)
    cb = CodebookConfig(init_method="random_normal", update_method="ema", health=CodebookHealthConfig(usage_regularization_weight=0.0, dead_code_restart=False))
    cfg = ModelConfig(hidden_dim=8, code_dim=4, num_codes=4, encoder_input=enc, codebook=cb)
    return VQArchetypeModel(feature_dim=2, config=cfg)


def _make_evaluator():
    cm = LobDepthCostModel(commission_rate=0.0001)
    align = RewardAlignment("paper_formula")

    def factory():
        return TradingEnv(cost_model=cm, reward_alignment=align, max_position=1)

    return Phase1Evaluator(replay_evaluator=Phase1ReplayEvaluator(env_factory=factory))


def test_evaluate_epoch_returns_required_metric_keys():
    evaluator = _make_evaluator()
    model = _make_model()
    records = [_record(f"r{i}") for i in range(4)]
    val_dataset = Phase1DemoDataset(records=records)
    out = evaluator.evaluate_epoch(
        epoch=0,
        model=model,
        val_data=val_dataset,
        val_records=records,
        full_validation=True,
    )
    required = {
        "reconstruction_accuracy",
        "code_usage_ratio",
        "perplexity",
        "switch_point_recall",
        "val_dp_teacher_net_return",
        "val_student_online_net_return",
        "val_return_capture_ratio",
        "val_sharpe_ratio",
        "val_max_drawdown",
        "inter_code_action_diversity",
        "decoder_sensitivity_to_code",
    }
    assert required.issubset(out.metrics.keys())


def test_fast_probe_uses_subset():
    evaluator = _make_evaluator()
    evaluator.fast_probe_size = 2
    model = _make_model()
    records = [_record(f"r{i}") for i in range(8)]
    val_dataset = Phase1DemoDataset(records=records)
    out = evaluator.evaluate_epoch(
        epoch=0, model=model, val_data=val_dataset, val_records=records, full_validation=False,
    )
    # per_horizon_replay_records 长度 == 2
    assert len(out.per_horizon_replay_records) == 2
