"""VQ Decoder — 根据状态和量化嵌入重建动作序列

# Section 4.1: Decoder
# p_θd(â_demo | s_demo, z_q)
# 根据状态序列和量化嵌入 z_q 生成动作 logits
#
# 本版本说明:
# 1. 回归论文接口：decoder 只依赖 (states, z_q)，不再吃 previous action。
# 2. 训练时 forward 直接输出逐时间步 logits，对应论文里的动作重建概率模型。
# 3. generate() 阶段额外加入“单次交易约束”的 constrained decoding，
#    用动态规划从逐步 logits 中解出最高分合法序列。
# 4. 这样既避免训练时 teacher forcing shortcut，又保持验证/推理时输出合法动作序列。
#
# 论文关联:
# - Section 4.1: archetype decoder
# - Algorithm 1: 单次交易约束只在解码阶段作为合法路径约束使用，不再作为训练输入 shortcut。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VQDecoder(nn.Module):
    """VQ 解码器，根据状态和量化嵌入生成动作序列。

    # Section 4.1: Decoder
    # 输入: states (batch, h, state_dim) 和 z_q (batch, code_dim)
    # 输出: 每个时间步的动作 logits
    #
    # 本版不再把 previous action / constraint state 作为训练输入。
    # 这是为了让 archetype code z_q 必须真正承担“整段行为模板”的条件作用，
    # 避免 decoder 主要依赖 teacher forcing 的上一动作去完成局部抄写。

    Args:
        state_dim: 状态向量维度 (默认 45)
        code_dim: 码本向量维度 (默认 16)
        hidden_dim: 隐藏层维度 (默认 128)
        action_dim: 动作空间大小 (默认 3)
    """

    FLAT_ACTION = 1

    def __init__(
        self,
        state_dim: int,
        code_dim: int = 16,
        hidden_dim: int = 128,
        action_dim: int = 3,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Section 4.1: Decoder
        # 这里保留“状态 + z_q 条件解码”的核心思想，
        # 但不再使用 autoregressive previous-action 输入。
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.code_to_gamma = nn.Linear(code_dim, hidden_dim)
        self.code_to_beta = nn.Linear(code_dim, hidden_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    # ------------------------------------------------------------------
    # 单次交易约束工具函数
    # ------------------------------------------------------------------

    def _compute_next_constraint(self, c: int, a_prev: int, a_next: int) -> int:
        """计算约束标志转移。

        该状态机与 DPPlanner._compute_next_constraint 保持一致，
        供 generate() 的 constrained decoding 使用。
        """
        prev_is_flat = (a_prev == self.FLAT_ACTION)
        next_is_flat = (a_next == self.FLAT_ACTION)

        if c == 0:
            if next_is_flat:
                return 0
            return 1

        if prev_is_flat and not next_is_flat:
            return -1
        if (not prev_is_flat) and (not next_is_flat) and a_prev != a_next:
            return -1
        return 1

    def _constrained_decode_single(self, logits: Tensor) -> Tensor:
        """对单条样本做合法路径动态规划解码。

        输入是逐时间步 logits；输出是在单次交易约束下分数最高的整段动作序列。
        这样可以把论文的 p(a_t | s_t, z_q) 与 Algorithm 1 的合法序列语法结合起来。
        """
        horizon, action_dim = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)

        # dp[t, a_prev, c] 表示处理完第 t-1 步后，处于 (a_prev, c) 的最高累计分数。
        neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
        dp = torch.full((horizon + 1, action_dim, 2), fill_value=neg_inf, device=logits.device, dtype=logits.dtype)
        back_action = torch.full((horizon, action_dim, 2), fill_value=-1, device=logits.device, dtype=torch.long)
        back_prev_a = torch.full((horizon, action_dim, 2), fill_value=-1, device=logits.device, dtype=torch.long)
        back_prev_c = torch.full((horizon, action_dim, 2), fill_value=-1, device=logits.device, dtype=torch.long)

        # 初始状态：上一动作是 flat，尚未交易。
        dp[0, self.FLAT_ACTION, 0] = 0.0

        for t in range(horizon):
            for prev_a in range(action_dim):
                for c in range(2):
                    prev_score = dp[t, prev_a, c]
                    if torch.isneginf(prev_score):
                        continue
                    for a_next in range(action_dim):
                        c_next = self._compute_next_constraint(c=int(c), a_prev=int(prev_a), a_next=int(a_next))
                        if c_next < 0:
                            continue
                        candidate = prev_score + log_probs[t, a_next]
                        if candidate > dp[t + 1, a_next, c_next]:
                            dp[t + 1, a_next, c_next] = candidate
                            back_action[t, a_next, c_next] = int(a_next)
                            back_prev_a[t, a_next, c_next] = int(prev_a)
                            back_prev_c[t, a_next, c_next] = int(c)

        # 终止：在所有合法终点状态里选累计分数最高者。
        terminal_scores = dp[horizon].reshape(-1)
        best_terminal = int(torch.argmax(terminal_scores).item())
        current_a = best_terminal // 2
        current_c = best_terminal % 2

        decoded: List[int] = []
        for t in range(horizon - 1, -1, -1):
            chosen_action = int(back_action[t, current_a, current_c].item())
            if chosen_action < 0:
                # 理论上不应发生；这里保底回退为 flat。
                chosen_action = self.FLAT_ACTION
                prev_a = self.FLAT_ACTION
                prev_c = 0
            else:
                prev_a = int(back_prev_a[t, current_a, current_c].item())
                prev_c = int(back_prev_c[t, current_a, current_c].item())
            decoded.append(chosen_action)
            current_a, current_c = prev_a, prev_c

        decoded.reverse()
        return torch.tensor(decoded, dtype=torch.long, device=logits.device)

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def forward(self, states: Tensor, z_q: Tensor, teacher_actions: Tensor | None = None) -> Tensor:
        """根据状态和量化嵌入生成动作 logits。

        teacher_actions 参数仅为兼容旧训练/验证接口保留，
        论文对齐版 forward 不再依赖它。
        """
        del teacher_actions

        # state 分支：逐时间步状态特征
        state_hidden = self.state_proj(states)
        state_hidden = self.state_norm(state_hidden)

        # z_q 分支：整段 archetype 条件，广播到每个时间步。
        gamma = self.code_to_gamma(z_q).unsqueeze(1)
        beta = self.code_to_beta(z_q).unsqueeze(1)

        # FiLM 式条件注入：仍然是 p(a | s, z_q)，但比简单拼接更能保证 z_q 真正参与决策。
        fused = torch.relu((1.0 + gamma) * state_hidden + beta)
        logits = self.output_mlp(fused)
        return logits

    def generate(self, states: Tensor, z_q: Tensor) -> Tuple[Tensor, Tensor]:
        """生成满足单次交易约束的动作序列。"""
        logits = self.forward(states=states, z_q=z_q, teacher_actions=None)
        batch = logits.shape[0]
        actions = []
        for b in range(batch):
            actions.append(self._constrained_decode_single(logits[b]))
        action_tensor = torch.stack(actions, dim=0)
        return action_tensor, logits
