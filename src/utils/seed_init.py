"""统一随机种子初始化，保证数据产物可复现。"""
from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy 和 Torch 随机源以保证数据产物可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
