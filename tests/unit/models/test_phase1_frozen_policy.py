"""Phase I frozen policy 单元测试。

测试用例:
- decoder 参数被冻结。
- decode_step() 可逐步输出 action logits。
- 修改未来 state 不改变过去 timestep 输出。
- decode() 仅诊断路径可用，主 replay 测试不调用。
"""
import pytest


class TestPhase1FrozenPolicy:

    def test_decoder_params_frozen(self):
        """decoder 参数全部 requires_grad=False。"""
        pass

    def test_decode_step_outputs_logits(self):
        """decode_step() 可逐步输出 action logits。"""
        pass

    def test_causal_invariant(self):
        """修改未来 state 不改变过去 timestep 输出。"""
        pass

    def test_decode_batch_diagnostic_only(self):
        """decode() 仅诊断路径可用。"""
        pass
