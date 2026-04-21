"""Phase I 工具函数模块

本模块提供 Phase I 训练中的通用工具函数。

Functions:
    set_reproducibility_seed: 设置随机种子
    compute_grad_norm: 计算梯度范数
    summarize_code_usage: 汇总 code 使用统计
"""

from typing import Any

import numpy as np
import torch


def set_reproducibility_seed(seed: int) -> None:
    """设置 Phase I 复现实验所需的随机种子。

    Args:
        seed: 随机种子值
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_grad_norm(parameters) -> float:
    """计算参数梯度的全局 L2 范数。

    Args:
        parameters: 模型参数迭代器

    Returns:
        梯度的 L2 范数
    """
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        norm = param.grad.detach().data.norm(2).item()
        total += norm * norm
    return float(total ** 0.5)


def summarize_code_usage(code_counts: np.ndarray) -> dict:
    """根据 epoch 内的 code 使用计数汇总 perplexity 与塌缩指标。

    Args:
        code_counts: 每个 code 的使用计数数组

    Returns:
        包含使用统计的字典，包括：
        - used_code_count: 被使用的 code 数量
        - dead_code_count: 未被使用的 code 数量
        - dominant_code_ratio: 最常用 code 的使用比例
        - codebook_entropy: codebook 熵
        - codebook_perplexity: codebook 困惑度
    """
    total = int(np.sum(code_counts))
    if total <= 0:
        return {
            "used_code_count": 0,
            "dead_code_count": int(len(code_counts)),
            "dominant_code_ratio": 0.0,
            "codebook_entropy": 0.0,
            "codebook_perplexity": 1.0,
        }

    probs = code_counts.astype(np.float64) / float(total)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs))) if probs.size else 0.0
    perplexity = float(np.exp(entropy)) if entropy > 0 else 1.0
    return {
        "used_code_count": int(np.sum(code_counts > 0)),
        "dead_code_count": int(np.sum(code_counts == 0)),
        "dominant_code_ratio": float(np.max(code_counts) / total),
        "codebook_entropy": entropy,
        "codebook_perplexity": perplexity,
    }