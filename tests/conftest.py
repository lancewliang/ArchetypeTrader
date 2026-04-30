"""Pytest 共享 fixture。

约定:
- 所有 fixture 都是确定性的（固定 seed），避免单元测试在 CI 上偶发抖动。
- 数据 fixture 优先生成内存级 `polars.DataFrame`，不写盘；
  集成测试需要真实文件时再使用 ``tmp_path``。
- 不依赖外部网络、外部数据库。
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

# Numpy / polars / torch 在没有依赖安装时也允许 import 失败；
# 各测试用例自身负责 import 并 skip。
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    """每个测试前重置 Python random / numpy 全局 seed，避免相互污染。"""
    random.seed(20260430)
    if np is not None:
        np.random.seed(20260430)


@pytest.fixture
def repo_root() -> Path:
    """仓库根目录，便于读取固定 fixture 文件。"""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def fixtures_dir(repo_root: Path) -> Path:
    """``tests/fixtures/`` 路径。fixture 数据由集成测试按需写入临时目录，
    本目录留给离线生成的小型 feather 数据使用。"""
    return repo_root / "tests" / "fixtures"
