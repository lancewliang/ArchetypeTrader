"""``Phase1DemoDataset`` 单元测试."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.data.data_augmentation import ContrastivePair
from src.data.dataset import Phase1DemoDataset, collate_phase1
from src.data.horizon_builder import HorizonRecord


def _record(sample_id: str, h: int = 4, rewards=None):
    rewards = rewards if rewards is not None else [0.0] * h
    return HorizonRecord(
        sample_id=sample_id,
        start_index=0,
        end_index=h - 1,
        pair="TEST",
        split="train",
        strata_label="up|low|mixed",
        states=[[0.0, 1.0] for _ in range(h)],
        prices=[100.0] * (h + 1),
        execution_books=[],
        actions=[1] * h,
        rewards=rewards,
    )


def test_dataset_length_matches_records():
    ds = Phase1DemoDataset(records=[_record("a"), _record("b")])
    assert len(ds) == 2


def test_getitem_returns_required_keys():
    ds = Phase1DemoDataset(records=[_record("a")])
    item = ds[0]
    assert set(item) == {
        "states",
        "actions",
        "rewards",
        "trajectory_return",
        "sample_id",
        "contrastive_pair_id",
    }
    assert item["states"].shape == (4, 2)
    assert item["actions"].dtype == torch.long


def test_pair_id_set_when_available():
    pair = ContrastivePair(pair_id="p0", sample_id_original="a", sample_id_shifted="a_sh", shift_bars=1)
    ds = Phase1DemoDataset(records=[_record("a")], contrastive_pairs=[pair])
    assert ds[0]["contrastive_pair_id"] == "p0"


def test_collate_stacks_tensors():
    ds = Phase1DemoDataset(records=[_record("a"), _record("b")])
    batch = collate_phase1([ds[0], ds[1]])
    assert batch["states"].shape == (2, 4, 2)
    assert batch["actions"].shape == (2, 4)
    assert batch["rewards"].shape == (2, 4)
    assert batch["trajectory_returns"].shape == (2,)
    assert batch["sample_ids"] == ["a", "b"]


def test_trajectory_return_uses_original_rewards_before_normalization():
    class _Normalizer:
        def transform(self, rewards):
            return [0.0 for _ in rewards]

    ds = Phase1DemoDataset(
        records=[_record("a", rewards=[1.0, 2.0, -0.5, 0.25])],
        reward_normalizer=_Normalizer(),
    )
    item = ds[0]
    assert torch.allclose(item["rewards"], torch.zeros(4))
    assert item["trajectory_return"] == pytest.approx(2.75)


def test_missing_actions_raises():
    rec = _record("a")
    rec.actions = None
    rec.rewards = None
    ds = Phase1DemoDataset(records=[rec])
    with pytest.raises(RuntimeError):
        ds[0]
