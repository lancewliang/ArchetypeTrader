"""统一日志工具 — ArchetypeTrader

提供统一的日志接口，支持 INFO/WARNING/ERROR 级别。
训练过程中输出损失值、奖励等关键指标。

需求: 7.6, 9.2
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """获取配置好的 logger 实例。

    返回带有控制台输出和统一格式的 logger。如果 logger 已有 handler，
    不会重复添加（避免多次调用产生重复日志）。

    Args:
        name: logger 名称，通常使用模块名如 ``__name__``。
        level: 日志级别，默认 ``logging.INFO``。

    Returns:
        配置好的 ``logging.Logger`` 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
