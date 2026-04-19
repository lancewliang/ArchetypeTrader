"""Progress bar helpers.

统一控制 tqdm 是否输出进度条，避免重定向日志时刷屏。
"""

from __future__ import annotations

import os
import sys
from typing import TextIO


def should_disable_tqdm(stream: TextIO | None = None) -> bool:
    """在非交互终端中禁用 tqdm，防止进度条写入日志文件。

    优先支持环境变量覆盖：
    - `ARCHETYPE_DISABLE_TQDM=1/true/yes/on` 强制禁用
    - `ARCHETYPE_DISABLE_TQDM=0/false/no/off` 强制启用
    """
    override = os.getenv("ARCHETYPE_DISABLE_TQDM", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False

    target_stream = stream if stream is not None else sys.stderr
    try:
        return not target_stream.isatty()
    except Exception:
        return True
