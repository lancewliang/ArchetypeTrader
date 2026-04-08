#!/usr/bin/env python
"""评估脚本 — 三阶段完整推理与指标计算

# 需求: 8.7, 8.8, 8.9
#
# 用法:
#   python scripts/evaluate.py
#   python scripts/evaluate.py --pair BTC
"""

import json
import os

import torch

from src.config import parse_args
from src.evaluation.inference_runner import evaluate_pair
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("评估开始，使用设备: %s", device)
    logger.info("结果目录批次: %s", config.train_batch_id)

    save_dir = os.path.join(config.get_batch_result_dir(), "evaluation")
    os.makedirs(save_dir, exist_ok=True)

    all_results = {}

    for pair in config.pairs:
        try:
            result = evaluate_pair(config, pair, device)
            all_results[pair] = result
            pair_save_dir = config.get_stage_result_dir(pair, "evaluation")
            os.makedirs(pair_save_dir, exist_ok=True)
            pair_path = os.path.join(pair_save_dir, f"{pair}_results.json")
            with open(pair_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info("结果已保存: %s", pair_path)
        except FileNotFoundError as e:
            logger.error("交易对 %s 评估失败: %s", pair, e)
            all_results[pair] = {"pair": pair, "error": str(e)}
        except Exception as e:
            logger.error("交易对 %s 评估异常: %s", pair, e)
            all_results[pair] = {"pair": pair, "error": str(e)}

    # 保存汇总结果
    summary_path = os.path.join(save_dir, "all_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("汇总结果已保存: %s", summary_path)

    # 打印汇总表格
    logger.info("=" * 70)
    logger.info(
        "%-6s %10s %10s %10s %10s %10s %10s",
        "Pair", "TR", "AVOL", "MDD", "ASR", "ACR", "ASoR",
    )
    logger.info("-" * 70)
    for pair, res in all_results.items():
        if "error" in res:
            logger.info("%-6s  ERROR: %s", pair, res["error"][:50])
        else:
            logger.info(
                "%-6s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f",
                pair,
                res["total_return"],
                res["annual_volatility"],
                res["max_drawdown"],
                res["annual_sharpe_ratio"],
                res["annual_calmar_ratio"],
                res["annual_sortino_ratio"],
            )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
