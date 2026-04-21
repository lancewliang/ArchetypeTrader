"""Phase II 通用工具函数模块

功能说明:
    提供随机种子设置、梯度范数计算等通用工具函数。
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_reproducibility_seed(seed: int) -> None:
    """设置 Phase II 复现实验所需的随机种子（与 Phase I 保持一致）。"""
    logger.info(
        "随机状态(设种子前): PYTHONHASHSEED=%s, torch.initial_seed=%d, numpy_state_head=%d, python_state_head=%d",
        os.getenv("PYTHONHASHSEED"),
        int(torch.initial_seed()),
        int(np.random.get_state()[1][0]),
        int(random.getstate()[1][0]),
    )

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(
        "随机状态(设种子后): seed=%d, torch.initial_seed=%d, numpy_state_head=%d, python_state_head=%d",
        int(seed),
        int(torch.initial_seed()),
        int(np.random.get_state()[1][0]),
        int(random.getstate()[1][0]),
    )


def parameter_grad_norm(parameters) -> float:
    """计算一组参数当前梯度的 L2 norm。

    功能说明:
        用于观察 PPO 更新时 policy / value 头是否真的在收到梯度，
        便于排查"critic 压过 actor"或"policy 基本没学动"的问题。
    """
    total_sq = 0.0
    has_grad = False
    for param in parameters:
        if param.grad is None:
            continue
        grad_norm = float(param.grad.detach().data.norm(2).item())
        total_sq += grad_norm * grad_norm
        has_grad = True
    if not has_grad:
        return 0.0
    return float(total_sq ** 0.5)
