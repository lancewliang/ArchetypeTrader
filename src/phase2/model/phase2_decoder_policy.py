"""Phase II frozen archetype decoder policy 骨架。

文件功能说明:
    本文件封装 Phase I 训练完成后的 VQ encoder-decoder 模型，只暴露
    ``code id -> base action sequence`` 的推理能力。Phase II selector 选择的是
    离散 archetype id；本 policy 负责把该 id 与当前 horizon states 送入冻结的
    Phase I decoder，得到可执行的基础动作序列。

设计边界:
    - 只封装冻结 Phase I decoder/codebook 推理入口；
    - 不训练 Phase I model，不更新 encoder、decoder 或 codebook 参数；
    - 不调用 Phase I encoder 生成 label；
    - 不计算交易 reward，不访问价格或盘口数据；
    - 不保存 checkpoint，也不负责 checkpoint selection。

使用场景:
    ``ArchetypeSelectionEnv`` 在执行 selector action 时调用 ``decode_actions()``；
    ``Phase2Evaluator`` 可以调用 ``decode_all_codes()`` 比较不同 archetype 的
    行为或收益；diagnostics 可调用 ``get_code_embeddings()`` 检查 codebook 向量。
"""

from __future__ import annotations

import torch

from ...model.tensor_data_types import (
    ArchetypeLabelTensor,
    LatentTensor,
)
from ...model.vq_archetype import ArchetypeVQModel


class FrozenArchetypeDecoderPolicy:
    """冻结 Phase I decoder 的 Phase II 推理封装。

    功能说明:
        持有一个已经加载好权重的 ``ArchetypeVQModel``，并把它切到 eval 模式、
        关闭梯度。对外只提供根据 selected code 解码动作、批量解码全部 code、
        读取 codebook embedding 的接口。

    设计边界:
        本类不理解 selector Q-network，也不计算 reward。它只负责把 Phase II
        选择出的 archetype id 转换为 Phase I decoder 可以消费的 latent 条件。

    使用场景:
        Phase II env、evaluator 和后续 Phase III refinement 需要复用 Phase I
        learned archetype decoder 时，统一通过本类调用。
    """

    def __init__(
        self,
        phase1_model: ArchetypeVQModel,
        device: torch.device | str,
    ) -> None:
        """冻结 Phase I model，只保留 decoder 推理能力。

        功能说明:
            保存 Phase I model 和运行设备，将模型移动到目标设备、切换到 eval 模式，
            并关闭所有参数梯度，确保 Phase II 训练不会更新 Phase I 产物。

        使用场景:
            ``Phase2MainFlow`` 加载 Phase I best checkpoint 后创建本 policy；
            ``ArchetypeSelectionEnv`` 和 ``Phase2Evaluator`` 复用同一个实例。

        参数:
            phase1_model: 已加载 best checkpoint 权重的 Phase I VQ 模型。
            device: decoder 推理设备，例如 ``"cuda"`` 或 ``"cpu"``。
        """

        self.phase1_model = phase1_model.to(device)
        self.device = torch.device(device)
        self.phase1_model.eval()
        for parameter in self.phase1_model.parameters():
            parameter.requires_grad_(False)

    def decode_actions(
        self,
        horizon_states: torch.Tensor,
        horizon_relative_states: torch.Tensor,
        horizon_trend_states: torch.Tensor,
        selected_code_ids: ArchetypeLabelTensor,
    ) -> torch.Tensor:
        """根据 horizon states 和 selected code ids 输出 base actions。

        功能说明:
            从 Phase I codebook 中取出 ``selected_code_ids`` 对应的 embeddings，
            按 horizon 时间步逐步调用冻结 decoder。第 ``tau`` 步只输入
            ``0..tau`` 的状态 prefix，并取 prefix 最后一步 logits 转成动作 id，
            避免接口层把未来状态传给当前步动作生成。

        使用场景:
            ``ArchetypeSelectionEnv.step()`` 执行 selector action 时调用，是 Phase II
            训练环境从 archetype id 到基础动作的主入口。

        参数:
            horizon_states: 当前 horizon 的完整状态序列，预期形状为
                ``[batch, horizon, state_dim]``。
            horizon_relative_states: 当前 horizon 的相对状态序列，预期形状为
                ``[batch, horizon, relative_state_dim]``。
            horizon_trend_states: 当前 horizon 的趋势状态序列，预期形状为
                ``[batch, horizon, trend_state_dim]``。
            selected_code_ids: selector 选择的 archetype id，预期形状为 ``[batch]``。

        返回:
            base action ids，预期形状为 ``[batch, horizon]``，动作语义与 Phase I
            decoder 一致：``0=short, 1=flat, 2=long``。
        """

        with torch.no_grad():
            states = horizon_states.to(self.device, dtype=torch.float32)
            relative_states = horizon_relative_states.to(self.device, dtype=torch.float32)
            trend_states = horizon_trend_states.to(self.device, dtype=torch.float32)
            code_ids = selected_code_ids.to(self.device, dtype=torch.long)
            z_q = self.get_code_embeddings(code_ids)
            return self._decode_actions_stepwise(
                states,
                relative_states,
                trend_states,
                z_q,
            )

    def decode_all_codes(
        self,
        horizon_states: torch.Tensor,
        horizon_relative_states: torch.Tensor,
        horizon_trend_states: torch.Tensor,
    ) -> torch.Tensor:
        """为每个样本解码全部 archetype 的 base actions。

        功能说明:
            对每个样本枚举 Phase I codebook 中的全部 code id，并复用逐步解码逻辑
            生成每个 code 对应的基础动作序列。

        使用场景:
            evaluator 计算 oracle best-code upper bound、诊断不同 code 的收益分布，
            或检查多个 archetype 是否输出高度重复动作时调用。

        参数:
            horizon_states: 当前 batch 的完整 horizon states，预期形状为
                ``[batch, horizon, state_dim]``。
            horizon_relative_states: 当前 batch 的完整 relative states，预期形状为
                ``[batch, horizon, relative_state_dim]``。
            horizon_trend_states: 当前 batch 的完整 trend states，预期形状为
                ``[batch, horizon, trend_state_dim]``。

        返回:
            base action ids，预期形状为 ``[batch, num_archetypes, horizon]``。
        """

        with torch.no_grad():
            states = horizon_states.to(self.device, dtype=torch.float32)
            relative_states = horizon_relative_states.to(self.device, dtype=torch.float32)
            trend_states = horizon_trend_states.to(self.device, dtype=torch.float32)
            if states.ndim != 3:
                raise ValueError(
                    "horizon_states must have shape [batch, horizon, state_dim]"
                )
            batch_size, horizon, _ = states.shape
            num_codes = int(self.phase1_model.num_archetypes)
            code_ids = torch.arange(num_codes, device=self.device, dtype=torch.long)
            code_ids = code_ids.unsqueeze(0).expand(batch_size, num_codes).reshape(-1)
            expanded_states = states.unsqueeze(1).expand(
                batch_size,
                num_codes,
                horizon,
                states.shape[-1],
            ).reshape(batch_size * num_codes, horizon, states.shape[-1])
            expanded_relative_states = relative_states.unsqueeze(1).expand(
                batch_size,
                num_codes,
                horizon,
                relative_states.shape[-1],
            ).reshape(batch_size * num_codes, horizon, relative_states.shape[-1])
            expanded_trend_states = trend_states.unsqueeze(1).expand(
                batch_size,
                num_codes,
                horizon,
                trend_states.shape[-1],
            ).reshape(batch_size * num_codes, horizon, trend_states.shape[-1])
            z_q = self.get_code_embeddings(code_ids)
            return self._decode_actions_stepwise(
                expanded_states,
                expanded_relative_states,
                expanded_trend_states,
                z_q,
            ).reshape(batch_size, num_codes, horizon)

    def get_code_embeddings(
        self,
        selected_code_ids: ArchetypeLabelTensor,
    ) -> LatentTensor:
        """从 Phase I codebook 读取 selected archetype embeddings。

        功能说明:
            后续实现应调用 Phase I quantizer 的 ``embedding_from_code()``，返回
            selected code ids 对应的 codebook vectors。

        使用场景:
            ``decode_actions()`` 内部取 decoder 条件向量时调用；diagnostics 也可用
            它检查 selected archetype embedding 的分布和相似度。

        参数:
            selected_code_ids: archetype id tensor，通常形状为 ``[batch]``。

        返回:
            codebook embedding tensor，形状为 ``[batch, latent_dim]``。
        """

        return self.phase1_model.quantizer.embedding_from_code(
            selected_code_ids.to(self.device, dtype=torch.long)
        )

    def _decode_actions_stepwise(
        self,
        states: torch.Tensor,
        relative_states: torch.Tensor,
        trend_states: torch.Tensor,
        z_q: LatentTensor,
    ) -> torch.Tensor:
        """逐步生成 base actions，每一步只暴露当前及历史 state prefix。"""

        self._validate_decode_inputs(
            states=states,
            relative_states=relative_states,
            trend_states=trend_states,
            z_q=z_q,
        )
        horizon = int(states.shape[1])
        action_steps: list[torch.Tensor] = []
        for step_index in range(horizon):
            prefix_end = step_index + 1
            logits = self.phase1_model.decoder(
                states[:, :prefix_end, :],
                relative_states[:, :prefix_end, :],
                trend_states[:, :prefix_end, :],
                z_q,
            )
            if logits.ndim != 3:
                raise ValueError(
                    "decoder logits must have shape [batch, prefix, action_dim], "
                    f"got {tuple(logits.shape)}"
                )
            expected_prefix_shape = (states.shape[0], prefix_end)
            if logits.shape[:2] != expected_prefix_shape:
                raise ValueError(
                    "decoder logits prefix shape must match input prefix, "
                    f"got {tuple(logits.shape[:2])}, expected {expected_prefix_shape}"
                )
            action_steps.append(torch.argmax(logits[:, -1, :], dim=-1))
        return torch.stack(action_steps, dim=1)

    @staticmethod
    def _validate_decode_inputs(
        *,
        states: torch.Tensor,
        relative_states: torch.Tensor,
        trend_states: torch.Tensor,
        z_q: LatentTensor,
    ) -> None:
        """校验 frozen decoder stepwise 推理的基础 batch/horizon 形状。"""

        if states.ndim != 3:
            raise ValueError("horizon_states must have shape [batch, horizon, state_dim]")
        if states.shape[1] <= 0:
            raise ValueError("horizon_states horizon length must be positive")
        if relative_states.ndim != 3:
            raise ValueError(
                "horizon_relative_states must have shape "
                "[batch, horizon, relative_state_dim]"
            )
        if trend_states.ndim != 3:
            raise ValueError(
                "horizon_trend_states must have shape [batch, horizon, trend_state_dim]"
            )
        if relative_states.shape[:2] != states.shape[:2]:
            raise ValueError("horizon relative states must share [batch, horizon]")
        if trend_states.shape[:2] != states.shape[:2]:
            raise ValueError("horizon trend states must share [batch, horizon]")
        if z_q.ndim != 2:
            raise ValueError("z_q must have shape [batch, latent_dim]")
        if z_q.shape[0] != states.shape[0]:
            raise ValueError("z_q batch size must match states batch size")
