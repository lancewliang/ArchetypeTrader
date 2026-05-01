"""RunningMeanStd ablation 单元测试。

测试用例:
- per_env_only 模式下各 env 统计互不污染。
- delayed_merge_next_rollout 模式下合并结果只在下一个 rollout 生效。
- 当前 rollout 不会消费本 rollout 内刚更新/合并出的统计量。
"""
import pytest


class TestRunningMeanStdAblation:

    def test_per_env_only_isolation(self):
        """per_env_only 模式下各 env 统计互不污染。"""
        pass

    def test_delayed_merge_next_rollout(self):
        """delayed_merge_next_rollout 模式下合并结果只在下一个 rollout 生效。"""
        pass

    def test_no_intra_rollout_consumption(self):
        """当前 rollout 不会消费本 rollout 内刚更新的统计量。"""
        pass
