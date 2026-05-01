"""Phase II streaming decode only 集成测试。

目标: 确保 HorizonEnv.step() 主路径只用 streaming decode。

断言:
- mock Phase1FrozenPolicy.decode()，若被调用则测试失败。
- decode_step() 被调用 h 次。
"""
import pytest


class TestPhase2StreamingDecodeOnly:

    @pytest.mark.integration
    def test_decode_not_called(self, tmp_path):
        """mock decode()，若被调用则测试失败。"""
        pass

    @pytest.mark.integration
    def test_decode_step_called_h_times(self, tmp_path):
        """decode_step() 被调用 h 次。"""
        pass
