"""GPU 显存日志与保护工具。"""

from __future__ import annotations

import logging

import torch


GPU_MEMORY_LIMIT_BYTES = 12 * 1024 ** 3


def _bytes_to_gib(num_bytes: int) -> float:
    return float(num_bytes) / float(1024 ** 3)


def reset_gpu_peak_memory_stats(device: torch.device) -> None:
    """在 CUDA 设备上重置 peak memory 统计。"""
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats(device)


def log_and_guard_gpu_memory(
    logger: logging.Logger,
    stage: str,
    device: torch.device,
    *,
    limit_bytes: int = GPU_MEMORY_LIMIT_BYTES,
    force_log: bool = False,
) -> None:
    """记录当前 GPU 显存；若超过阈值则终止程序。

    这里同时检查 allocated 和 reserved。
    - allocated 更接近当前活跃 tensor 占用
    - reserved 更接近 `nvidia-smi` 里常看到的 PyTorch 进程占用
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    max_allocated = int(torch.cuda.max_memory_allocated(device))
    max_reserved = int(torch.cuda.max_memory_reserved(device))
    observed = max(allocated, reserved)

    if force_log or observed > limit_bytes:
        logger.info(
            "[GPU] %s | allocated=%.2f GiB, reserved=%.2f GiB, "
            "max_allocated=%.2f GiB, max_reserved=%.2f GiB, limit=%.2f GiB",
            stage,
            _bytes_to_gib(allocated),
            _bytes_to_gib(reserved),
            _bytes_to_gib(max_allocated),
            _bytes_to_gib(max_reserved),
            _bytes_to_gib(limit_bytes),
        )

    if observed > limit_bytes:
        logger.error(
            "GPU 显存超过阈值，终止程序: stage=%s, allocated=%.2f GiB, reserved=%.2f GiB, limit=%.2f GiB",
            stage,
            _bytes_to_gib(allocated),
            _bytes_to_gib(reserved),
            _bytes_to_gib(limit_bytes),
        )
        raise SystemExit(1)
