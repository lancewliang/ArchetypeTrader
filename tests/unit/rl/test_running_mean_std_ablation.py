"""RunningMeanStd ablation 单元测试。"""
from __future__ import annotations

import numpy as np

from src.rl.running_mean_std import RunningMeanStdAblationManager


class TestRunningMeanStdAblation:

    def test_per_env_only_isolation(self):
        """per_env_only 模式下各 env 统计互不污染。"""
        mgr = RunningMeanStdAblationManager(2, (1,), "per_env_only")
        mgr.observe(0, np.array([[10.0], [12.0]]))
        mgr.observe(1, np.array([[-10.0], [-12.0]]))
        assert mgr.env_stats[0].mean[0] > 0
        assert mgr.env_stats[1].mean[0] < 0

    def test_delayed_merge_next_rollout(self):
        """delayed_merge_next_rollout 模式下合并结果只在下一个 rollout 生效。"""
        mgr = RunningMeanStdAblationManager(2, (1,), "delayed_merge_next_rollout")
        before = mgr.normalize(0, np.array([10.0]))[0]
        mgr.observe(0, np.array([[10.0], [10.0]]))
        during = mgr.normalize(0, np.array([10.0]))[0]
        mgr.finalize_rollout()
        after = mgr.normalize(0, np.array([10.0]))[0]
        assert during == before
        assert abs(after) < abs(before)

    def test_no_intra_rollout_consumption(self):
        """当前 rollout 不会消费本 rollout 内刚更新的统计量。"""
        mgr = RunningMeanStdAblationManager(1, (1,), "delayed_merge_next_rollout")
        mgr.observe(0, np.array([[100.0]]))
        assert mgr.active_stats.mean[0] == 0.0
        assert mgr.pending_stats.mean[0] > 0.0
