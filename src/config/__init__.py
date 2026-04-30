"""Phase I 配置入口。

仅暴露 ``Phase1Config``。所有子配置（数据、DP、模型、训练、评估、selection guardrail）
都集中在 ``phase1_config.py``，避免散落到多文件造成 import cycle。
"""

from .phase1_config import Phase1Config  # noqa: F401
