#!/usr/bin/env python
"""评估脚本 — 三阶段完整推理与指标计算

# 需求: 8.7, 8.8, 8.9
#
# 用法:
#   python scripts/evaluate.py                             # 默认 test split
#   python scripts/evaluate.py --split val                 # 仅 val（Phase II 后对比用）
#   python scripts/evaluate.py --split val test            # val + test 两个 split
#   python scripts/evaluate.py --pair BTC --split test
#   python scripts/evaluate.py --stage-label phase2_eval --split val test
"""

import argparse
import json
import os
import sys

import torch

from src.config import parse_args
from src.evaluation.inference_runner import evaluate_pair, evaluate_pair_dp
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _pop_evaluate_args(argv: list[str]) -> tuple[list[str], list[str], str]:
    """从 argv 中摘出 --split、--stage-label、--with-dp，返回 (splits, stage_label, with_dp, remaining_argv)。"""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--split", nargs="+", default=["test"],
                     choices=["val", "test"], dest="splits")
    pre.add_argument("--stage-label", type=str, default=None, dest="stage_label",
                     help="输出子目录前缀，用于区分 phase2_eval / phase3_eval")
    pre.add_argument("--with-dp", action="store_true", default=False, dest="with_dp",
                     help="同时在相同 split 上运行 DP 基准，输出对比表")
    known, remaining = pre.parse_known_args(argv)
    return known.splits, known.stage_label, known.with_dp, remaining


def main() -> None:
    splits, stage_label, with_dp, remaining_argv = _pop_evaluate_args(sys.argv[1:])

    config = parse_args(remaining_argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("评估开始，使用设备: %s", device)
    logger.info("结果目录批次: %s", config.train_batch_id)
    logger.info("评估 split: %s, stage_label: %s, with_dp: %s", splits, stage_label or "(默认)", with_dp)

    save_dir = os.path.join(config.get_batch_result_dir(), "evaluation")
    os.makedirs(save_dir, exist_ok=True)

    all_results: dict = {}

    for split in splits:
        logger.info("=" * 60)
        logger.info("开始评估 split=%s", split)
        logger.info("=" * 60)
        split_results: dict = {}

        # 输出子目录：有 stage_label 时用 "{stage_label}_{split}"，否则用 "evaluation_{split}"
        subdir = f"{stage_label}_{split}" if stage_label else f"evaluation_{split}"

        # phase2_eval 时不加载 phase3 模型，phase3_eval（或无 label）时加载
        with_phase3 = stage_label != "phase2_eval"

        for pair in config.pairs:
            try:
                result = evaluate_pair(config, pair, device, split=split,
                                       output_subdir=subdir, with_phase3=with_phase3)
                split_results[pair] = result
                pair_save_dir = config.get_stage_result_dir(pair, subdir)
                os.makedirs(pair_save_dir, exist_ok=True)
                pair_path = os.path.join(pair_save_dir, f"{pair}_results.json")
                with open(pair_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info("[%s/%s] 结果已保存: %s", stage_label or "eval", split, pair_path)
            except FileNotFoundError as e:
                logger.error("[%s] 交易对 %s 评估失败: %s", split, pair, e)
                split_results[pair] = {"pair": pair, "split": split, "error": str(e)}
            except Exception as e:
                logger.error("[%s] 交易对 %s 评估异常: %s", split, pair, e)
                split_results[pair] = {"pair": pair, "split": split, "error": str(e)}

        all_results[split] = split_results

        # 保存本 split 汇总
        summary_name = f"all_results_{stage_label}_{split}.json" if stage_label else f"all_results_{split}.json"
        summary_path = os.path.join(save_dir, summary_name)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(split_results, f, indent=2, ensure_ascii=False)
        logger.info("[%s] 汇总结果已保存: %s", split, summary_path)

        # 打印模型汇总表格
        logger.info("=" * 70)
        logger.info("stage=%-12s split=%-4s  %6s %10s %10s %10s %10s %10s %10s",
                    stage_label or "eval", split, "Pair", "TR", "AVOL", "MDD", "ASR", "ACR", "ASoR")
        logger.info("-" * 70)
        for pair, res in split_results.items():
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

        # DP 基准对比
        if with_dp:
            dp_subdir = f"dp_{split}"
            logger.info("运行 DP 基准 [split=%s] ...", split)
            dp_results: dict = {}
            for pair in config.pairs:
                try:
                    dp_res = evaluate_pair_dp(config, pair, device, split=split, output_subdir=dp_subdir)
                    dp_results[pair] = dp_res
                except Exception as e:
                    logger.error("[DP/%s] 交易对 %s 异常: %s", split, pair, e)
                    dp_results[pair] = {"pair": pair, "split": split, "mode": "dp", "error": str(e)}

            # 对比表
            logger.info("=" * 90)
            logger.info("%-6s  %-14s %10s %10s %10s %10s  (split=%s)",
                        "Pair", "Mode", "TR", "AVOL", "MDD", "ASR", split)
            logger.info("-" * 90)
            for pair in config.pairs:
                model_res = split_results.get(pair, {})
                dp_res = dp_results.get(pair, {})
                model_ok = "error" not in model_res
                dp_ok = "error" not in dp_res
                model_tr = model_res.get("total_return", float("nan"))
                dp_tr = dp_res.get("total_return", float("nan"))
                gap = model_tr - dp_tr if (model_ok and dp_ok) else float("nan")
                if model_ok:
                    logger.info("%-6s  %-14s %10.4f %10.4f %10.4f %10.4f",
                                pair, stage_label or "model",
                                model_res["total_return"], model_res["annual_volatility"],
                                model_res["max_drawdown"], model_res["annual_sharpe_ratio"])
                if dp_ok:
                    logger.info("%-6s  %-14s %10.4f %10.4f %10.4f %10.4f",
                                pair, "dp",
                                dp_res["total_return"], dp_res["annual_volatility"],
                                dp_res["max_drawdown"], dp_res["annual_sharpe_ratio"])
                if model_ok and dp_ok:
                    logger.info("%-6s  %-14s %10.4f  (model - dp)",
                                pair, "gap(TR)", gap)
                logger.info("-" * 90)
            logger.info("=" * 90)


if __name__ == "__main__":
    main()
