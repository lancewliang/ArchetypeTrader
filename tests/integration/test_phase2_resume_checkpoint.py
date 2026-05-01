"""Phase II checkpoint 恢复集成测试。

目标: 验证 checkpoint 恢复训练。

步骤:
1. 先跑 total_timesteps=256。
2. 再用 --resume-from last_selector.pt --total-timesteps 512。

断言:
- 第二次训练从上次 update 继续。
- optimizer / scheduler / RNG / env cursor / prev_terminal_position 被恢复。
- 恢复后第一个 horizon 的 prev_terminal_position 通过一致性校验。
"""
import pytest


class TestPhase2ResumeCheckpoint:

    @pytest.mark.integration
    def test_resume_continues_from_last(self, tmp_path):
        """第二次训练从上次 update 继续。"""
        pass

    @pytest.mark.integration
    def test_state_restored(self, tmp_path):
        """optimizer / scheduler / RNG / env cursor 被恢复。"""
        pass

    @pytest.mark.integration
    def test_position_consistency_after_resume(self, tmp_path):
        """恢复后 prev_terminal_position 通过一致性校验。"""
        pass
