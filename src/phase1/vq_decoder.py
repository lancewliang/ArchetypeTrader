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
        """推理时使用: 在 logits 上施加 single-trade 约束（全向量化实现）。

        对 MLP 输出的 (h, action_dim) logits，搜索最优的 single-change
        分割点，使得动作序列为 "action_a × t + action_b × (h-t)" 的形式，
        且总 log-probability 最大。

        DP 轨迹的结构: 整个 horizon 最多一次动作变化。
        - 无变化: 全程同一个动作 (change_point = h)
        - 一次变化: 前 t 步动作 a，后 h-t 步动作 b (a ≠ b)

        搜索复杂度: O(h × action_dim²)，对 h=72, action_dim=3 可忽略。
        全部在 GPU tensor 上完成，无 Python 循环和 .item() 调用。

        Args:
            states: (batch, h, state_dim)
            z_q: (batch, code_dim)

        Returns:
            actions: (batch, h) 符合 single-trade 约束的动作序列
        """
        logits = self.forward(states, z_q)  # (batch, h, action_dim)
        log_probs = torch.log_softmax(logits, dim=-1)  # (batch, h, action_dim)

        batch, h, num_actions = log_probs.shape
        device = logits.device

        # 前缀和: prefix_sum[:, t, a] = Σ_{i=0}^{t} log_prob[:, i, a]
        prefix_sum = torch.cumsum(log_probs, dim=1)  # (batch, h, action_dim)

        # suffix_sum[:, t, a] = Σ_{i=t}^{h-1} log_prob[:, i, a]
        suffix_sum = prefix_sum[:, -1:, :] - prefix_sum + log_probs  # (batch, h, action_dim)

        # ---- 情况 1: 无变化，全程同一个动作 ----
        # no_change_scores[:, a] = prefix_sum[:, h-1, a]
        no_change_scores = prefix_sum[:, -1, :]  # (batch, num_actions)
        no_change_best_score, no_change_best_a = no_change_scores.max(dim=1)  # (batch,)

        # ---- 情况 2: 在第 t 步变化 (前 t 步 a1, 后 h-t 步 a2, a1 ≠ a2) ----
        if h > 1:
            # 对 t=1..h-1, 计算 prefix_sum[:, t-1, a1] + suffix_sum[:, t, a2] 对所有 (a1, a2) 对
            # prefix_part: (batch, h-1, num_actions) — 取 t=0..h-2 即 prefix_sum[:, 0:h-1, :]
            prefix_part = prefix_sum[:, :h-1, :]  # (batch, h-1, num_actions)
            # suffix_part: (batch, h-1, num_actions) — 取 t=1..h-1 即 suffix_sum[:, 1:h, :]
            suffix_part = suffix_sum[:, 1:, :]  # (batch, h-1, num_actions)

            # combined[:, t, a1, a2] = prefix_part[:, t, a1] + suffix_part[:, t, a2]
            # shape: (batch, h-1, num_actions, num_actions)
            combined = prefix_part.unsqueeze(-1) + suffix_part.unsqueeze(-2)

            # 屏蔽 a1 == a2 的对角线（设为 -inf 使其不会被选中）
            diag_mask = torch.eye(num_actions, dtype=torch.bool, device=device)  # (A, A)
            combined.masked_fill_(diag_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # 在 (t, a1, a2) 三个维度上找全局最大值
            # 先展平为 (batch, (h-1)*A*A)，再 argmax
            flat = combined.reshape(batch, -1)  # (batch, (h-1)*A*A)
            flat_best_score, flat_best_idx = flat.max(dim=1)  # (batch,)

            # 从 flat index 恢复 (t_idx, a1, a2)
            a2_size = num_actions
            a1_size = num_actions
            best_a2_change = flat_best_idx % a2_size
            best_a1_change = (flat_best_idx // a2_size) % a1_size
            best_t_change = flat_best_idx // (a1_size * a2_size)  # 0-indexed → 对应 t=1..h-1
            best_t_change = best_t_change + 1  # 转为实际 change point (1..h-1)

            # ---- 合并两种情况，选最优 ----
            use_change = flat_best_score > no_change_best_score  # (batch,)
        else:
            # h=1: 只有一步，不可能有 change point
            use_change = torch.zeros(batch, dtype=torch.bool, device=device)
            best_a1_change = no_change_best_a
            best_a2_change = no_change_best_a
            best_t_change = torch.full_like(no_change_best_a, h)

        best_a1 = torch.where(use_change, best_a1_change, no_change_best_a)  # (batch,)
        best_a2 = torch.where(use_change, best_a2_change, no_change_best_a)  # (batch,)
        best_t = torch.where(use_change, best_t_change, torch.full_like(best_t_change, h))  # (batch,)

        # ---- 向量化填充动作序列 ----
        # time_indices: (1, h) — [0, 1, ..., h-1]
        time_indices = torch.arange(h, device=device).unsqueeze(0)  # (1, h)
        # best_t: (batch, 1)
        best_t_expanded = best_t.unsqueeze(1)  # (batch, 1)
        # 前 best_t 步填 best_a1，后面填 best_a2
        best_actions = torch.where(
            time_indices < best_t_expanded,
            best_a1.unsqueeze(1).expand(batch, h),
            best_a2.unsqueeze(1).expand(batch, h),
        )

        return best_actions
