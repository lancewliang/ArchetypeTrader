"""运行时初始化工具。"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path


class RuntimeUtils:
    """集中管理训练入口的运行时初始化。

    功能:
        1. 初始化日志，统一 console/file handler 和日志格式。
        2. 初始化随机种子，覆盖 Python、NumPy 和可选 PyTorch。

    设计原则:
        该类不依赖 Phase I 具体业务对象，可被后续 Phase II/III 或数据处理脚本
        复用。NumPy/PyTorch 使用延迟导入，避免没有安装相关库时影响纯工具调用。
    """

    DEFAULT_LOG_FORMAT = (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

    @classmethod
    def init_logging(
        cls,
        *,
        name: str = "archetype_trader",
        log_file: str | Path | None = None,
        level: int | str = logging.INFO,
        reset_handlers: bool = True,
    ) -> logging.Logger:
        """初始化并返回指定名称的 logger。

        参数:
            name: logger 名称。Phase I 可使用 ``archetype_trader.phase1``。
            log_file: 可选日志文件路径。传入时会创建父目录并写入同一份日志。
            level: 日志级别，支持 ``logging.INFO`` 或 ``"INFO"`` 这类字符串。
            reset_handlers: 是否清理已有 handler。默认清理，避免重复调用时同一
                条日志被打印多次。

        返回:
            已完成 console/file handler 配置的 ``logging.Logger``。
        """

        resolved_level = cls._resolve_log_level(level)
        logger = logging.getLogger(name)
        logger.setLevel(resolved_level)
        logger.propagate = False

        if reset_handlers:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)

        formatter = logging.Formatter(
            fmt=cls.DEFAULT_LOG_FORMAT,
            datefmt=cls.DEFAULT_DATE_FORMAT,
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(resolved_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    @classmethod
    def init_random_seed(
        cls,
        seed: int,
        *,
        deterministic: bool = True,
    ) -> dict[str, bool | int | str | None]:
        """初始化 Python、NumPy 和 PyTorch 随机种子。

        参数:
            seed: 随机种子。必须是非负整数。
            deterministic: 为 True 时，若 PyTorch 可用，会尽量启用确定性算法并
                关闭 cuDNN benchmark，提升训练复现性。

        返回:
            初始化状态摘要。调用方可写入 metrics/report，便于复现实验。
        """

        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        os.environ["PYTHONHASHSEED"] = str(seed)
        cublas_workspace_config = None
        if deterministic:
            cublas_workspace_config = cls.configure_cublas_workspace()
        random.seed(seed)

        numpy_seeded = cls._try_seed_numpy(seed)
        torch_seeded = cls._try_seed_torch(seed, deterministic=deterministic)

        return {
            "seed": seed,
            "python_seeded": True,
            "numpy_seeded": numpy_seeded,
            "torch_seeded": torch_seeded,
            "deterministic": deterministic,
            "cublas_workspace_config": cublas_workspace_config,
        }

    @classmethod
    def configure_cublas_workspace(cls) -> str:
        """Ensure deterministic CUDA matmul uses a supported CuBLAS workspace."""

        return os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG",
            cls.DEFAULT_CUBLAS_WORKSPACE_CONFIG,
        )

    @staticmethod
    def _resolve_log_level(level: int | str) -> int:
        if isinstance(level, int):
            return level

        normalized = level.upper()
        resolved = logging.getLevelName(normalized)
        if isinstance(resolved, int):
            return resolved

        raise ValueError(f"unknown log level: {level}")

    @staticmethod
    def _try_seed_numpy(seed: int) -> bool:
        try:
            import numpy as np
        except ImportError:
            return False

        np.random.seed(seed)
        return True

    @staticmethod
    def _try_seed_torch(seed: int, *, deterministic: bool) -> bool:
        try:
            import torch
        except ImportError:
            return False

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

        return True
