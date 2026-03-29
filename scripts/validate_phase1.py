#!/usr/bin/env python
"""Phase I 独立验证脚本。

用法:
  python scripts/validate_phase1.py --pair ETH
"""

import torch

from src.config import parse_args
from src.phase1.validation import validate_phase1_artifacts
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """加载已有的 dp_trajectories 与 phase1_model，生成验证报告。"""
    config = parse_args()
    pair = config.pairs[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = validate_phase1_artifacts(
        config=config,
        pair=pair,
        device=device,
        dp_check_limit=256,
    )
    logger.info("Phase I 验证完成: overall_passed=%s", report["status"]["overall_passed"])
    if report["status"]["hard_failures"]:
        logger.error("Phase I 验证硬失败: %s", report["status"]["hard_failures"])
    if report["status"]["soft_warnings"]:
        logger.warning("Phase I 验证软告警: %s", report["status"]["soft_warnings"])


if __name__ == "__main__":
    main()
