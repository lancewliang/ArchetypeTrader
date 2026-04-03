"""VQ Decoder — MLP 动作序列解码器 + Single-Trade 推理约束

# Section 4.1: Decoder
# p_θd(â_demo | s_demo, z_q)
#
# 训练时: MLP 逐步独立预测 action logits，用标准 CrossEntropyLoss 训练。
#   MLP 对 short/flat/long 的区分能力好，diversity 好。
#
# 推理时: 在 MLP 输出的 logits 上施加 single-trade 约束后处理，
#   从 logits 中搜索最优的 single-change-point 分割，
#   把"嘈杂"的逐步预测整理成符合 DP 轨迹结构的动作序列。
#   这解决了 MLP 频繁切换动作导致执行成本爆炸的问题。
"""

import torch
import torch.nn as nn
from torch import Tensor


class VQDecoder(nn.Module):
    """MLP 解码器 + single-trade 推理约束。

    Args:
        state_dim: 状态向量维度 (默认 45)
        code_dim: 码本向量维度 (默认 16)
        hidden_dim: 隐藏层维度 (默认 128)
        action_dim: 动作空间大小 (默认 3)
    """

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

        self.mlp = nn.Sequential(
            nn.Linear(state_dim + code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states: Tensor, z_q: Tensor) -> Tensor:
        """生成 action logits（训练和推理共用）。

        Args:
            states: (batch, h, state_dim)
            z_q: (batch, code_dim)

        Returns:
            action_logits: (batch, h, action_dim)
        """
        batch, h, _ = states.shape
        z_q_expanded = z_q.unsqueeze(1).expand(batch, h, self.code_dim)
        decoder_input = torch.cat([states, z_q_expanded], dim=-1)
        return self.mlp(decoder_input)

    @torch.no_grad()
    def decode_with_single_trade_constraint(
        self, states: Tensor, z_q: Tensor,
    ) -> Tensor:
        """推理时使用: 在 logits 上施加 single-trade 约束。

        对 MLP 输出的 (h, action_dim) logits，搜索最优的 single-change
        分割点，使得动作序列为 "action_a × t + action_b × (h-t)" 的形式，
        且总 log-probability 最大。

        DP 轨迹的结构: 整个 horizon 最多一次动作变化。
        - 无变化: 全程同一个动作 (change_point = h)
        - 一次变化: 前 t 步动作 a，后 h-t 步动作 b (a ≠ b)

        搜索复杂度: O(h × action_dim²)，对 h=72, action_dim=3 可忽略。

        Args:
            states: (batch, h, state_dim)
            z_q: (batch, code_dim)

        Returns:
            actions: (batch, h) 符合 single-trade 约束的动作序列
        """
        logits = self.forward(states, z_q)  # (batch, h, action_dim)
        log_probs = torch.log_softmax(logits, dim=-1)  # (batch, h, action_dim)

        batch, h, num_actions = log_probs.shape

        # 前缀和: prefix_sum[t, a] = Σ_{i=0}^{t-1} log_prob[i, a]
        # 即前 t 步全选动作 a 的总 log-probability
        prefix_sum = torch.cumsum(log_probs, dim=1)  # (batch, h, action_dim)

        # suffix_sum[t, a] = Σ_{i=t}^{h-1} log_prob[i, a]
        # 即从第 t 步到末尾全选动作 a 的总 log-probability
        suffix_sum = prefix_sum[:, -1:, :] - prefix_sum + log_probs  # (batch, h, action_dim)

        best_actions = torch.zeros(batch, h, dtype=torch.long, device=logits.device)

        for b in range(batch):
            best_score = float("-inf")
            best_a1 = 0
            best_a2 = 0
            best_t = h  # change point (h 表示无变化)

            # 情况 1: 无变化，全程同一个动作
            for a in range(num_actions):
                score = prefix_sum[b, h - 1, a].item()
                if score > best_score:
                    best_score = score
                    best_a1 = a
                    best_a2 = a
                    best_t = h

            # 情况 2: 在第 t 步变化 (前 t 步 a1, 后 h-t 步 a2, a1 ≠ a2)
            for t in range(1, h):
                for a1 in range(num_actions):
                    pre_score = prefix_sum[b, t - 1, a1].item()
                    for a2 in range(num_actions):
                        if a2 == a1:
                            continue
                        post_score = suffix_sum[b, t, a2].item()
                        score = pre_score + post_score
                        if score > best_score:
                            best_score = score
                            best_a1 = a1
                            best_a2 = a2
                            best_t = t

            # 填充动作序列
            best_actions[b, :best_t] = best_a1
            best_actions[b, best_t:] = best_a2

        return best_actions
