"""VQ encoder-decoder 整体模型组装.

设计文档锚点: §6.1 / §6.3 / §6.7。

强约束: ``ArchetypeDecoder`` 必须 causal (单向 LSTM)；任何 bidirectional / 全 horizon
pooling / 未来 state pooling 都会破坏 Phase II/III 在线推理因果性。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from src.phase1.config import ModelConfig

from .encoder_inputs import EncoderInputAdapter
from .vector_quantizer import QuantizeOutput, VectorQuantizer


def _no_grad():
    if torch is None:  # pragma: no cover
        def decorator(fn):
            return fn
        return decorator
    return torch.no_grad()


@dataclass
class ModelOutputs:
    action_logits: "torch.Tensor"
    z_e: "torch.Tensor"
    z_q: "torch.Tensor"
    z_q_no_grad: "torch.Tensor"
    code_id: Optional["torch.Tensor"]


class ArchetypeEncoder(nn.Module if nn is not None else object):  # type: ignore[misc]
    """单向 LSTM encoder。

    输入: 适配后的 ``[batch, h, fusion_dim]``。
    输出: ``z_e ∈ R^{code_dim}``。

    实现注意:
    - encoder 同时保留 train/eval 一致行为；不使用 bidirectional 让推理更简单。
    - 取最后一步 hidden state 后经 MLP 投影到 ``code_dim``。
    - 论文层面 encoder 可双向（仅离线 label 分配用），但本实现保持单向以便
      latent 可视化探针在 train/val 上行为一致。
    """

    def __init__(self, fusion_dim: int, hidden_dim: int, code_dim: int) -> None:
        if nn is None:  # pragma: no cover
            raise ImportError("ArchetypeEncoder 需要 torch")
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, code_dim),
        )

    def forward(self, fused_inputs):
        """``fused_inputs`` 形状 ``[batch, h, fusion_dim]``，返回 ``z_e: [batch, code_dim]``。"""
        out, _ = self.lstm(fused_inputs)
        last = out[:, -1, :]
        return self.proj(last)


class ArchetypeDecoder(nn.Module if nn is not None else object):  # type: ignore[misc]
    """因果 decoder。

    输入: ``[batch, h, hidden_dim + code_dim]``，``z_q`` 在每个 timestep 拼接。
    输出: ``[batch, h, 3]`` action logits。

    严格约束（论文与设计 §6.7）:
    - LSTM 必须 ``bidirectional=False``。
    - 第 ``τ`` 步 logits 只能依赖 ``s_{0:τ}`` 与选中的 ``z_q``，不能依赖 ``s_{τ+1:}``。
    - 不允许使用 ``sequence_summary`` / 全 horizon pooling / 未来收益作为 decoder 输入。
    - 单元测试 ``test_modifying_future_states_does_not_change_past_logits`` 验证该不变量。
    """

    def __init__(
        self,
        feature_dim: int,
        code_dim: int,
        hidden_dim: int,
    ) -> None:
        if nn is None:  # pragma: no cover
            raise ImportError("ArchetypeDecoder 需要 torch")
        super().__init__()
        self.state_proj = nn.Linear(feature_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim + code_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,  # 因果约束
        )
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, states, z_q):
        """``states``: [B, h, F]，``z_q``: [B, code_dim]。

        Steps
        -----
        1. ``state_h = self.state_proj(states)`` → ``[batch, h, hidden_dim]``。
        2. ``cond = z_q.unsqueeze(1).expand(-1, h, -1)``: 每 timestep 拼接同一个 code。
        3. LSTM 单向递推，输出 ``logits = self.head(...) ∈ [batch, h, 3]``。

        因果性: 由于 LSTM 单向 + 输入仅来自 (state, z_q)（与未来无关），
        修改 ``states[τ+1:]`` 不会改变 ``logits[:, :τ+1, :]``。
        """
        b, h, _ = states.shape
        state_h = self.state_proj(states)
        cond = z_q.unsqueeze(1).expand(-1, h, -1)
        x = torch.cat([state_h, cond], dim=-1)
        out, _ = self.lstm(x)
        return self.head(out)


class VQArchetypeModel(nn.Module if nn is not None else object):  # type: ignore[misc]
    """整体模型: input adapter + encoder + VQ + decoder。"""

    def __init__(self, feature_dim: int, config: ModelConfig) -> None:
        if nn is None:  # pragma: no cover
            raise ImportError("VQArchetypeModel 需要 torch")
        super().__init__()
        self.config = config
        self.input_adapter = EncoderInputAdapter(feature_dim, config.encoder_input)
        self.encoder = ArchetypeEncoder(
            fusion_dim=config.encoder_input.fusion_dim,
            hidden_dim=config.hidden_dim,
            code_dim=config.code_dim,
        )
        self.quantizer = VectorQuantizer(
            num_codes=config.num_codes,
            code_dim=config.code_dim,
            config=config.codebook,
        )
        self.decoder = ArchetypeDecoder(
            feature_dim=feature_dim,
            code_dim=config.code_dim,
            hidden_dim=config.hidden_dim,
        )

    def forward(self, states, actions, rewards) -> ModelOutputs:
        """完整前向。

        Steps
        -----
        1. ``fused = self.input_adapter(states, actions, rewards)``。
        2. ``z_e = self.encoder(fused)``。
        3. ``q = self.quantizer.quantize(z_e)``。
        4. ``logits = self.decoder(states, q.z_q)``。

        Returns
        -------
        ModelOutputs : 含 ``action_logits / z_e / z_q (STE) / z_q_no_grad / code_id``。
        """
        fused = self.input_adapter(states, actions, rewards)
        z_e = self.encoder(fused)
        q: QuantizeOutput = self.quantizer.quantize(z_e)
        logits = self.decoder(states, q.z_q)
        return ModelOutputs(
            action_logits=logits,
            z_e=z_e,
            z_q=q.z_q,
            z_q_no_grad=q.z_q_no_grad,
            code_id=q.code_id,
        )

    def forward_pretrain(self, states, actions, rewards) -> ModelOutputs:
        """Phase A 前向：跳过 VQ，用 encoder latent 直接条件化 decoder。"""
        fused = self.input_adapter(states, actions, rewards)
        z_e = self.encoder(fused)
        logits = self.decoder(states, z_e)
        return ModelOutputs(
            action_logits=logits,
            z_e=z_e,
            z_q=z_e,
            z_q_no_grad=z_e.detach(),
            code_id=None,
        )

    @_no_grad()
    def encode(self, states, actions, rewards):
        """仅编码到 ``(code_id, z_e)``，不跑 decoder。

        用于 horizon labels 导出与 Phase II selector 监督；
        ``@torch.no_grad`` 保证不会意外构建反向图。
        """
        fused = self.input_adapter(states, actions, rewards)
        z_e = self.encoder(fused)
        q = self.quantizer.quantize(z_e)
        return q.code_id, z_e

    def decode(self, states, code_id):
        """Phase II/III 推理路径: 不再走 encoder，直接按 ``code_id`` 取 codebook。

        - ``code_id``: ``[B]`` long。
        - 返回 ``[B, h, 3]`` action logits。
        - 不带 ``@torch.no_grad`` 是因为 Phase II selector 训练时仍可能对
          decoder 走 forward（但 decoder 已 frozen → grad 自然为 None）。
        """
        z_q = self.quantizer.codebook[code_id]
        return self.decoder(states, z_q)
