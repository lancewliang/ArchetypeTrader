"""Feather/Arrow IPC 与 JSON 的底层 IO 帮助函数。

设计文档锚点: §4.17 ``src/utils/feather_io.py``。

约束:
- 仅做 IO 适配（``polars.read_ipc`` / ``DataFrame.write_ipc``、原子写、shard 合并）。
- 不做 schema 校验、不引用业务字段、不维护全局状态。
- 所有 ``write_*`` 必须原子: 先写 ``*.tmp``，``os.replace`` 重命名后再返回。

为什么必须原子写: 训练过程中如果 OOM/SIGKILL 发生在写文件中途，会留下半写的 feather；
下游 evaluator/trainer 可能读到 corrupted 文件然后默默给出错误结果。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Union

PathLike = Union[str, "os.PathLike[str]"]


def read_ipc(path: PathLike):
    """读取 Feather/Arrow IPC 文件为 ``polars.DataFrame``。"""
    import polars as pl

    return pl.read_ipc(path)


def write_ipc(frame, path: PathLike) -> Path:
    """原子写 Feather/Arrow IPC。

    实现思路: 在目标目录创建临时文件，写完后 ``os.replace`` 重命名；
    防止程序崩溃时留下半写文件被下游误读。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # 在同目录建临时文件，保证 os.replace 跨设备时不会失败。
    with tempfile.NamedTemporaryFile(
        delete=False, dir=target.parent, suffix=".tmp"
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        frame.write_ipc(tmp_path)
        os.replace(tmp_path, target)
    except Exception:
        # 写失败时清理临时文件，避免留垃圾。
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return target


def read_csv_for_debug(path: PathLike):
    """仅供 fixture / 调试使用的 CSV 读取通道。

    主路径必须使用 Feather；如果调用方在生产代码中误用 CSV，
    应在 schema 校验阶段被拦下。
    """
    import polars as pl

    return pl.read_csv(path)


def atomic_write_json(payload: Any, path: PathLike, *, indent: int = 2) -> Path:
    """原子写 JSON，保留可读缩进，便于审计 ``phase1_report.json``。

    缩进固定为 2，sort_keys=True，避免不同 epoch 写出的 diff 抖动。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=target.parent,
        suffix=".tmp",
        encoding="utf-8",
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=indent, sort_keys=True)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, target)
    return target


def read_json(path: PathLike) -> Any:
    """读取 JSON。简单封装，统一文件不存在/解析失败的报错语义。"""
    target = Path(path)
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records: Iterable[dict], path: PathLike) -> Path:
    """原子写 JSONL，用于 failure case 错题本与 reject_event 流式日志。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=target.parent,
        suffix=".tmp",
        encoding="utf-8",
    ) as tmp:
        for rec in records:
            tmp.write(json.dumps(rec, ensure_ascii=False, sort_keys=True))
            tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, target)
    return target


def read_jsonl(path: PathLike) -> List[dict]:
    """读取 JSONL；空行直接跳过，不当作错误（兼容追加写入场景）。"""
    out: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def merge_ipc_shards(shard_paths: Iterable[PathLike], target: PathLike) -> Path:
    """合并多个 Feather shard 为单文件。

    用途:
    - DP 并行生成 demonstration 后合并成 ``demos_train.feather``。
    - latent snapshot 多 epoch 合并。

    实现注意:
    - shard 必须按调用方传入顺序拼接，不重新排序，保证 deterministic。
    - 合并后用 ``write_ipc`` 原子写，避免半写。
    """
    import polars as pl

    paths = list(shard_paths)
    if not paths:
        raise ValueError("merge_ipc_shards: 至少需要一个 shard")
    frames = [pl.read_ipc(p) for p in paths]
    merged = pl.concat(frames, how="vertical")
    return write_ipc(merged, target)
