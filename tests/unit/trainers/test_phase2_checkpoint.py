"""Phase II checkpoint 单元测试。

测试用例:
- 保存 last_selector.pt。
- verdict 允许时 promote best_selector.pt。
- manifest 写入 metrics、reasons、hash。
"""
import pytest
import torch

from src.trainers.phase2_checkpoint import Phase2CheckpointManager
from src.trainers.phase2_selection_policy import Phase2SelectionVerdict


@pytest.fixture
def ckpt_mgr(tmp_path):
    return Phase2CheckpointManager(tmp_path)


@pytest.fixture
def dummy_state():
    model = torch.nn.Linear(4, 2)
    return {"model_state": model.state_dict(), "update_count": 0}


class TestPhase2CheckpointManager:

    def test_save_last(self, ckpt_mgr, dummy_state):
        """保存 last_selector.pt。"""
        path = ckpt_mgr.save_last(dummy_state, update_idx=0)
        assert path.exists()

    def test_promote_best(self, ckpt_mgr, dummy_state):
        """verdict 允许时 promote best_selector.pt。"""
        ckpt_mgr.save_last(dummy_state, update_idx=0)
        verdict = Phase2SelectionVerdict(decision="promote_to_best")
        entry = ckpt_mgr.commit_verdict(dummy_state, verdict, 0, {"val_net_return": 1.0})
        assert entry.is_best is True
        assert ckpt_mgr.best_path.exists()

    def test_manifest_records(self, ckpt_mgr, dummy_state):
        """manifest 写入 metrics、reasons、hash。"""
        ckpt_mgr.save_last(dummy_state, update_idx=0)
        verdict = Phase2SelectionVerdict(
            decision="reject", reasons=["max_drawdown exceeded"]
        )
        entry = ckpt_mgr.commit_verdict(dummy_state, verdict, 0, {"mdd": 0.5})
        assert entry.verdict == "reject"
        assert "max_drawdown exceeded" in entry.reasons
        assert ckpt_mgr.manifest_path.exists()

    def test_replay_log_saved(self, ckpt_mgr):
        """replay_log_last_complete_checkpoint.feather 被保存。"""
        records = [
            {"update_idx": 0, "env_id": 0, "sample_id": "s0",
             "chosen_code": 1, "final_position": 0, "reward_raw": 0.1,
             "boundary_cost": 0.01, "risk_triggered": False}
        ]
        path = ckpt_mgr.save_replay_log(records)
        assert path.exists()

    def test_load_checkpoint(self, ckpt_mgr, dummy_state):
        """加载 checkpoint。"""
        ckpt_mgr.save_last(dummy_state, update_idx=0)
        loaded = ckpt_mgr.load(ckpt_mgr.last_path)
        assert "model_state" in loaded

    def test_multiple_promotes_only_one_best(self, ckpt_mgr, dummy_state):
        """多次 promote 后只有最后一个 is_best=True。"""
        ckpt_mgr.save_last(dummy_state, update_idx=0)
        v1 = Phase2SelectionVerdict(decision="promote_to_best")
        ckpt_mgr.commit_verdict(dummy_state, v1, 0, {})
        v2 = Phase2SelectionVerdict(decision="promote_to_best")
        ckpt_mgr.commit_verdict(dummy_state, v2, 1, {})
        best_entries = [e for e in ckpt_mgr._entries if e.is_best]
        assert len(best_entries) == 1
        assert best_entries[0].update_idx == 1
