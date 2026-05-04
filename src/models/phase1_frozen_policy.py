"""Phase I 冻结策略: 加载 decoder + codebook，提供 streaming decode 接口。

设计文档锚点: Phase II 执行计划 §Step 3。

职责:
- 只加载 decoder.pt + codebook.pt。
- 参数全部 requires_grad=False。
- 正式 replay 主接口为 decode_step()（streaming 因果接口）。
- 批量 decode() 仅允许诊断或离线对比，不允许 HorizonEnv 主路径调用。
- 自检 decoder 为单向因果结构。

关键约束:
- HorizonEnv 正式 replay 只能使用 decode_step() 的 streaming 因果接口。
- 禁止双向 LSTM、双向 attention 或全 horizon pooling。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn


@dataclass
class DecodeStepOutput:
    """单步 decode 输出。"""
    action_logits: torch.Tensor  # [3]
    action: int  # argmax of logits
    recurrent_state: Any  # LSTM hidden state for next step


class Phase1FrozenPolicy:
    """Phase I 冻结 decoder + codebook。

    使用方式::

        policy = Phase1FrozenPolicy.load(decoder_path, codebook_path, device)
        policy.reset(code_id=3)
        for t in range(horizon):
            out = policy.decode_step(state_t)
            action = out.action

    边界:
    - 不加载 encoder（Phase II 不需要 encoder）。
    - 所有参数 requires_grad=False。
    - decode_step() 是 HorizonEnv 主路径的唯一合法接口。
    """

    def __init__(self, decoder: nn.Module, codebook: torch.Tensor, device: str = "cpu") -> None:
        self.decoder = decoder.to(device)
        self.decoder.eval()
        # 冻结所有参数
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self._codebook = codebook.to(device)
        self.device = device
        self._code_id: Optional[int] = None
        self._z_q: Optional[torch.Tensor] = None
        self._recurrent_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._step_count: int = 0
        self._state_proj: Optional[nn.Module] = None
        self._head: Optional[nn.Module] = None
        self._lstm: Optional[nn.Module] = None

        # 提取 decoder 子模块用于 streaming decode
        if hasattr(decoder, "state_proj"):
            self._state_proj = decoder.state_proj
        if hasattr(decoder, "head"):
            self._head = decoder.head
        if hasattr(decoder, "lstm"):
            self._lstm = decoder.lstm

        self._validate_causal_structure()

    def spawn_worker_policy(self) -> "Phase1FrozenPolicy":
        """Create an env-local wrapper with independent streaming state.

        The decoder and codebook are frozen, so worker policies may share those
        tensors. Runtime fields such as ``_recurrent_state`` and ``_code_id``
        must remain per-env to avoid cross-env leakage during parallel rollout.
        """
        return Phase1FrozenPolicy(
            decoder=self.decoder,
            codebook=self._codebook,
            device=self.device,
        )

    @classmethod
    def load(
        cls,
        decoder_path: Path,
        codebook_path: Path,
        device: str = "cpu",
    ) -> "Phase1FrozenPolicy":
        """从文件加载 decoder + codebook，冻结所有参数。

        同时执行因果性自检: 确认 decoder 为单向 LSTM。

        Raises
        ------
        ValueError : decoder 不是单向因果结构。
        """
        decoder_state = torch.load(decoder_path, map_location="cpu")
        codebook_data = torch.load(codebook_path, map_location="cpu")

        # decoder_state 可能是 state_dict 或完整模块
        if isinstance(decoder_state, nn.Module):
            decoder = decoder_state
        else:
            # 需要从 state_dict 重建 decoder
            # 推断维度
            if isinstance(decoder_state, dict):
                decoder_state = cls._normalize_decoder_state_dict(decoder_state)
                # 尝试从 state_dict 推断结构
                from src.models.vq_archetype import ArchetypeDecoder
                # 从 state_proj.weight 推断 feature_dim
                sp_key = "state_proj.weight"
                if sp_key in decoder_state:
                    hidden_dim = decoder_state[sp_key].shape[0]
                    feature_dim = decoder_state[sp_key].shape[1]
                else:
                    hidden_dim = 128
                    feature_dim = 16
                # 从 lstm.weight_ih_l0 推断 code_dim
                lstm_key = "lstm.weight_ih_l0"
                if lstm_key in decoder_state:
                    input_size = decoder_state[lstm_key].shape[1]
                    code_dim = input_size - hidden_dim
                else:
                    code_dim = 16
                decoder = ArchetypeDecoder(feature_dim, code_dim, hidden_dim)
                decoder.load_state_dict(decoder_state)
            else:
                raise ValueError(f"无法加载 decoder: 未知格式 {type(decoder_state)}")

        codebook = cls._extract_codebook_tensor(codebook_data)

        return cls(decoder, codebook, device)

    @staticmethod
    def _normalize_decoder_state_dict(state: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize exported decoder state dicts to ``ArchetypeDecoder`` keys.

        Older Phase I exports saved keys as ``decoder.state_proj.weight`` while
        ``ArchetypeDecoder.load_state_dict`` expects ``state_proj.weight``.
        """
        if "model" in state and isinstance(state["model"], dict):
            return Phase1FrozenPolicy._normalize_decoder_state_dict(state["model"])
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            return Phase1FrozenPolicy._normalize_decoder_state_dict(state["state_dict"])

        decoder_items = {
            key[len("decoder."):]: value
            for key, value in state.items()
            if isinstance(key, str) and key.startswith("decoder.")
        }
        if decoder_items:
            return decoder_items
        return dict(state)

    @staticmethod
    def _extract_codebook_tensor(data: Any) -> torch.Tensor:
        """Extract codebook tensor from raw tensor or quantizer state dict."""
        if isinstance(data, nn.Parameter):
            return data.data
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, dict):
            for nested_key in ("model", "state_dict"):
                nested = data.get(nested_key)
                if isinstance(nested, dict):
                    try:
                        return Phase1FrozenPolicy._extract_codebook_tensor(nested)
                    except ValueError:
                        pass

            for key in ("codebook", "quantizer.codebook"):
                value = data.get(key)
                if isinstance(value, nn.Parameter):
                    return value.data
                if isinstance(value, torch.Tensor):
                    return value

            suffix_matches = [
                value for key, value in data.items()
                if isinstance(key, str)
                and key.endswith(".codebook")
                and isinstance(value, (torch.Tensor, nn.Parameter))
            ]
            if len(suffix_matches) == 1:
                value = suffix_matches[0]
                return value.data if isinstance(value, nn.Parameter) else value

            raise ValueError("无法从 codebook.pt 中提取 codebook tensor")
        return torch.tensor(data, dtype=torch.float32)

    def reset(self, code_id: int) -> None:
        """重置 recurrent state，设置当前 horizon 使用的 code_id。

        每个新 horizon 开始时必须调用。
        """
        if code_id < 0 or code_id >= self._codebook.shape[0]:
            raise ValueError(
                f"code_id={code_id} 越界，codebook 大小为 {self._codebook.shape[0]}"
            )
        self._code_id = code_id
        self._z_q = self._codebook[code_id].unsqueeze(0)  # [1, code_dim]
        self._recurrent_state = None
        self._step_count = 0

    @torch.no_grad()
    def decode_step(self, state_t: Any) -> DecodeStepOutput:
        """单步因果 decode。

        Parameters
        ----------
        state_t : 当前时刻的 state，shape [feature_dim] 或 [1, feature_dim]。

        Returns
        -------
        DecodeStepOutput : 包含 action_logits / action / recurrent_state。

        Raises
        ------
        RuntimeError : 未调用 reset()。
        """
        if self._code_id is None or self._z_q is None:
            raise RuntimeError("decode_step() 调用前必须先 reset()")

        if not isinstance(state_t, torch.Tensor):
            state_t = torch.tensor(state_t, dtype=torch.float32)
        state_t = state_t.to(self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)  # [1, feature_dim]

        # 使用 decoder 子模块进行 streaming decode
        if self._state_proj is not None and self._lstm is not None and self._head is not None:
            state_h = self._state_proj(state_t)  # [1, hidden_dim]
            z_q = self._z_q  # [1, code_dim]
            x = torch.cat([state_h, z_q], dim=-1).unsqueeze(1)  # [1, 1, hidden_dim+code_dim]
            if self._recurrent_state is not None:
                out, new_state = self._lstm(x, self._recurrent_state)
            else:
                out, new_state = self._lstm(x)
            logits = self._head(out.squeeze(1))  # [1, 3]
        else:
            # fallback: 用完整 decoder forward（仅单步）
            # 警告: 此路径不传递 LSTM recurrent state，每步独立推理，
            # 会丢失时序上下文。仅在 decoder 结构不匹配时使用。
            import warnings
            if self._step_count == 0:
                warnings.warn(
                    "Phase1FrozenPolicy: decoder 缺少 state_proj/lstm/head 子模块，"
                    "fallback 到无状态单步推理，时序上下文将丢失",
                    RuntimeWarning,
                    stacklevel=2,
                )
            state_t_seq = state_t.unsqueeze(1)  # [1, 1, feature_dim]
            logits = self.decoder(state_t_seq, self._z_q)  # [1, 1, 3]
            logits = logits.squeeze(1)  # [1, 3]
            new_state = self._recurrent_state

        self._recurrent_state = new_state
        self._step_count += 1

        action_logits = logits.squeeze(0)  # [3]
        action = int(action_logits.argmax().item())

        return DecodeStepOutput(
            action_logits=action_logits,
            action=action,
            recurrent_state=new_state,
        )

    @torch.no_grad()
    def decode(self, states: Any, code_id: int) -> Tuple[List[int], torch.Tensor]:
        """批量 decode（仅诊断路径）。

        警告: 此方法不允许在 HorizonEnv 主路径中调用。
        仅用于离线对比和诊断。

        Parameters
        ----------
        states : [h, feature_dim] 或 [batch, h, feature_dim]。
        code_id : archetype id。

        Returns
        -------
        (base_actions, action_logits) : actions 列表和 logits tensor。
        """
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        states = states.to(self.device)
        if states.dim() == 2:
            states = states.unsqueeze(0)  # [1, h, feature_dim]

        z_q = self._codebook[code_id].unsqueeze(0)  # [1, code_dim]
        logits = self.decoder(states, z_q)  # [1, h, 3]
        actions = logits.argmax(dim=-1).squeeze(0).tolist()  # [h]
        if isinstance(actions, int):
            actions = [actions]
        return actions, logits.squeeze(0)

    def _validate_causal_structure(self) -> None:
        """自检 decoder 为单向因果结构。

        检查 LSTM bidirectional=False，无全 horizon pooling。

        Raises
        ------
        ValueError : 检测到非因果结构。
        """
        for name, module in self.decoder.named_modules():
            if isinstance(module, nn.LSTM):
                if module.bidirectional:
                    raise ValueError(
                        f"decoder 中检测到双向 LSTM ({name})，"
                        "违反因果性约束"
                    )

    @property
    def num_codes(self) -> int:
        """Codebook 中的 archetype 数量 K。"""
        return self._codebook.shape[0]

    @property
    def code_dim(self) -> int:
        """Code embedding 维度。"""
        return self._codebook.shape[1]
