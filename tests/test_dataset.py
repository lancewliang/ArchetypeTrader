"""TrajectoryDataset 单元测试"""

import numpy as np
import pytest
import torch

from src.data.dataset import TrajectoryDataset


# ---- 固定参数 ----
N, H, STATE_DIM = 50, 72, 45


def _make_arrays(n=N, h=H, state_dim=STATE_DIM):
    """生成合法的 numpy 测试数据。"""
    states = np.random.randn(n, h, state_dim).astype(np.float32)
    actions = np.random.randint(0, 3, size=(n, h)).astype(np.int64)
    rewards = np.random.randn(n, h).astype(np.float32)
    return states, actions, rewards


# ---- 基本功能 ----


class TestTrajectoryDatasetBasic:
    def test_len(self):
        ds = TrajectoryDataset(*_make_arrays())
        assert len(ds) == N

    def test_getitem_shapes(self):
        ds = TrajectoryDataset(*_make_arrays())
        s, a, r = ds[0]
        assert s.shape == (H, STATE_DIM)
        assert a.shape == (H,)
        assert r.shape == (H,)

    def test_getitem_dtypes(self):
        ds = TrajectoryDataset(*_make_arrays())
        s, a, r = ds[0]
        assert s.dtype == torch.float32
        assert a.dtype == torch.int64
        assert r.dtype == torch.float32

    def test_content_preserved(self):
        """确保 numpy → tensor 转换不改变数值。"""
        states, actions, rewards = _make_arrays(n=5, h=10, state_dim=8)
        ds = TrajectoryDataset(states, actions, rewards)
        s, a, r = ds[2]
        np.testing.assert_allclose(s.numpy(), states[2], atol=1e-7)
        np.testing.assert_array_equal(a.numpy(), actions[2])
        np.testing.assert_allclose(r.numpy(), rewards[2], atol=1e-7)

    def test_single_sample(self):
        ds = TrajectoryDataset(*_make_arrays(n=1))
        assert len(ds) == 1
        s, a, r = ds[0]
        assert s.shape == (H, STATE_DIM)


# ---- 输入校验 ----


class TestTrajectoryDatasetValidation:
    def test_mismatched_n(self):
        states = np.zeros((10, H, STATE_DIM))
        actions = np.zeros((11, H), dtype=np.int64)
        rewards = np.zeros((10, H))
        with pytest.raises(ValueError, match="样本数不一致"):
            TrajectoryDataset(states, actions, rewards)

    def test_mismatched_horizon(self):
        states = np.zeros((N, H, STATE_DIM))
        actions = np.zeros((N, H + 1), dtype=np.int64)
        rewards = np.zeros((N, H))
        with pytest.raises(ValueError, match="horizon 长度不一致"):
            TrajectoryDataset(states, actions, rewards)

    def test_states_wrong_ndim(self):
        with pytest.raises(ValueError, match="3D"):
            TrajectoryDataset(
                np.zeros((N, STATE_DIM)),
                np.zeros((N, H), dtype=np.int64),
                np.zeros((N, H)),
            )

    def test_actions_wrong_ndim(self):
        with pytest.raises(ValueError, match="2D"):
            TrajectoryDataset(
                np.zeros((N, H, STATE_DIM)),
                np.zeros((N, H, 3), dtype=np.int64),
                np.zeros((N, H)),
            )

    def test_rewards_wrong_ndim(self):
        with pytest.raises(ValueError, match="2D"):
            TrajectoryDataset(
                np.zeros((N, H, STATE_DIM)),
                np.zeros((N, H), dtype=np.int64),
                np.zeros((N,)),
            )


# ---- .npz 加载 ----


class TestTrajectoryDatasetFromNpz:
    def test_load_from_npz(self, tmp_path):
        states, actions, rewards = _make_arrays(n=20, h=10, state_dim=8)
        npz_path = tmp_path / "traj.npz"
        np.savez(str(npz_path), states=states, actions=actions, rewards=rewards)

        ds = TrajectoryDataset.from_npz(npz_path)
        assert len(ds) == 20
        s, a, r = ds[0]
        assert s.shape == (10, 8)
        np.testing.assert_allclose(s.numpy(), states[0], atol=1e-7)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="轨迹文件不存在"):
            TrajectoryDataset.from_npz("/nonexistent/path.npz")

    def test_missing_key(self, tmp_path):
        npz_path = tmp_path / "bad.npz"
        np.savez(str(npz_path), states=np.zeros((2, 3, 4)))
        with pytest.raises(KeyError, match="缺少必要的键"):
            TrajectoryDataset.from_npz(npz_path)


# ---- DataLoader 兼容性 ----


class TestTrajectoryDatasetDataLoader:
    def test_dataloader_batch(self):
        ds = TrajectoryDataset(*_make_arrays(n=16, h=10, state_dim=8))
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        s_batch, a_batch, r_batch = batch
        assert s_batch.shape == (4, 10, 8)
        assert a_batch.shape == (4, 10)
        assert r_batch.shape == (4, 10)
