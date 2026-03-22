"""Adaptive Layer Normalization (AdaLN) 模块

# Section 4.3: Adaptive Layer Normalization
# 用 s_ref2 条件化 s_ref1 的处理
# AdaLN(x, c) = γ(c) * LayerNorm(x) + β(c)
#
# 条件向量 c (s_ref2) 通过线性投影生成 scale γ 和 shift β，
# 对 LayerNorm 归一化后的特征进行仿射变换，实现市场状态的条件化。
"""

import torch
import torch.nn as nn
from torch import Tensor


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization (AdaLN).

    # Section 4.3: Adaptive Layer Normalization
    # 使用条件向量 c 对输入特征 x 进行条件化归一化：
    # AdaLN(x, c) = γ(c) × LayerNorm(x) + β(c)

    Args:
        feature_dim: 被条件化的特征维度 (对应 s_ref1 经过编码后的维度)
        condition_dim: 条件向量维度 (对应 s_ref2 = [e_a_sel, a_base, R_arche, τ_remain])
    """

    def __init__(self, feature_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(feature_dim)
        # γ(c): 条件化缩放参数
        self.gamma_proj = nn.Linear(condition_dim, feature_dim)
        # β(c): 条件化偏移参数
        self.beta_proj = nn.Linear(condition_dim, feature_dim)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """对输入特征进行条件化归一化。

        Args:
            x: (batch, feature_dim) 市场观测特征
            condition: (batch, condition_dim) 条件向量

        Returns:
            (batch, feature_dim) 条件化后的特征
        """
        # Section 4.3: AdaLN(x, c) = γ(c) × LayerNorm(x) + β(c)
        normalized = self.layer_norm(x)
        gamma = self.gamma_proj(condition)
        beta = self.beta_proj(condition)
        output = gamma * normalized + beta
        return output
