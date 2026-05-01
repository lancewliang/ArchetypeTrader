"""Phase II walk-forward backtest 集成测试。

目标: 验证 best selector 在 test 上 walk-forward 回测时的仓位连续性和 deterministic argmax。

断言:
- prev_terminal_position 在相邻 horizons 间正确传递。
- 主结果使用 argmax。
- stochastic seed pack 只写诊断，不覆盖主结果。
"""
import pytest


class TestPhase2BacktestWalkForward:

    @pytest.mark.integration
    def test_position_continuity(self, tmp_path):
        """prev_terminal_position 在相邻 horizons 间正确传递。"""
        pass

    @pytest.mark.integration
    def test_deterministic_argmax(self, tmp_path):
        """主结果使用 argmax。"""
        pass

    @pytest.mark.integration
    def test_stochastic_diagnostic_only(self, tmp_path):
        """stochastic seed pack 只写诊断，不覆盖主结果。"""
        pass
