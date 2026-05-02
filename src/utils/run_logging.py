"""Run-scoped file logging helpers."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple


_RUN_HANDLER_FLAG = "_archetype_run_file_handler"


def configure_run_logger(
    *,
    phase: str,
    pair: str,
    batch_id: str,
    log_root: str | Path = "logs",
) -> Tuple[logging.Logger, Path]:
    """Create a phase logger writing to ``logs/{pair}/{batch_id}/ts-*.log``.

    Existing run file handlers for the same phase logger are closed first. This
    keeps sequential runs in one Python process, such as ablations, from writing
    into each other's files.
    """
    logger = logging.getLogger(f"archetype.{phase}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, _RUN_HANDLER_FLAG, False):
            logger.removeHandler(handler)
            handler.close()

    log_dir = Path(log_root) / pair / batch_id
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"ts-{timestamp}-{os.getpid()}.log"
    suffix = 1
    while log_path.exists():
        log_path = log_dir / f"ts-{timestamp}-{os.getpid()}-{suffix}.log"
        suffix += 1

    handler = logging.FileHandler(log_path, encoding="utf-8")
    setattr(handler, _RUN_HANDLER_FLAG, True)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.info("log_file=%s", log_path)
    return logger, log_path
