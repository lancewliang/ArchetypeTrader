"""Phase II dataset 单元测试。

测试用例:
- state 维度与 feature_columns + position_encoding + optional extensions 一致。
- position_continuity=true 时 prev_terminal_position 必须进入状态。
- 数据集不重写 Phase I horizon slicing 语义。
- phase2_dataset 不调用 DP。
"""
import pytest


class TestPhase2Dataset:
    """Phase II dataset 测试。"""

    def test_state_dim_matches_spec(self):
        """state 维度与 feature_columns + position_encoding 一致。"""
        # TODO: 实现
        pass

    def test_position_continuity_includes_prev_position(self):
        """position_continuity=true 时 prev_terminal_position 进入状态。"""
        # TODO: 实现
        pass

    def test_no_horizon_slicing_rewrite(self):
        """数据集不重写 Phase I horizon slicing 语义。"""
        # TODO: 实现
        pass

    def test_no_dp_call(self):
        """phase2_dataset 不调用 DP。"""
        # TODO: 实现
        pass
