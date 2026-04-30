"""动作分类与切换点指标.

设计文档锚点: §4.11。

这些函数都设计为纯函数，输入为 numpy ndarray / list 之类的标准结构，便于测试。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import math


@dataclass
class ConfusionMatrix:
    """3x3 混淆矩阵: row=true (0/1/2), col=pred。"""
    matrix: List[List[int]]

    def per_class(self) -> Dict[str, Dict[str, float]]:
        """每类 precision / recall / f1。"""
        out: Dict[str, Dict[str, float]] = {}
        names = ["short", "flat", "long"]
        for i in range(3):
            tp = self.matrix[i][i]
            fp = sum(self.matrix[r][i] for r in range(3)) - tp
            fn = sum(self.matrix[i][c] for c in range(3)) - tp
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            out[names[i]] = {"precision": precision, "recall": recall, "f1": f1}
        return out


def _argmax(logits_row: Sequence[float]) -> int:
    best_i = 0
    best_v = logits_row[0]
    for i in range(1, len(logits_row)):
        if logits_row[i] > best_v:
            best_v = logits_row[i]
            best_i = i
    return best_i


def _flatten_pred(logits) -> List[List[int]]:
    """logits ``[B, h, 3]`` → 每个样本的预测序列 ``[B, h]``。"""
    out: List[List[int]] = []
    for batch_row in logits:
        out.append([_argmax(step) for step in batch_row])
    return out


def reconstruction_accuracy(logits, actions) -> float:
    """``mean(argmax(logits) == actions)``。

    Parameters
    ----------
    logits : ``[B, h, 3]`` Python list/ndarray。
    actions : ``[B, h]`` Python list/ndarray。

    Notes
    -----
    设计 §9.5: 该指标只能作为 sanity check，不能单独决定 best；
    flat 占比过高时容易虚高。配套使用 ``weighted_reconstruction_accuracy``
    与 ``non_flat_accuracy``。
    """
    preds = _flatten_pred(logits)
    correct = 0
    total = 0
    for p_row, a_row in zip(preds, actions):
        for p, a in zip(p_row, a_row):
            total += 1
            if p == a:
                correct += 1
    return correct / max(total, 1)


def weighted_reconstruction_accuracy(
    logits, actions, class_weights: Dict[int, float]
) -> float:
    """按 class weight 加权后的 accuracy。"""
    preds = _flatten_pred(logits)
    weight_sum = 0.0
    score_sum = 0.0
    for p_row, a_row in zip(preds, actions):
        for p, a in zip(p_row, a_row):
            w = class_weights.get(int(a), 1.0)
            weight_sum += w
            if p == a:
                score_sum += w
    return score_sum / max(weight_sum, 1e-9)


def non_flat_accuracy(logits, actions) -> float:
    """只在 ``actions != 1`` 处计算 accuracy。"""
    preds = _flatten_pred(logits)
    correct = 0
    total = 0
    for p_row, a_row in zip(preds, actions):
        for p, a in zip(p_row, a_row):
            if a == 1:
                continue
            total += 1
            if p == a:
                correct += 1
    if total == 0:
        return 0.0
    return correct / total


def action_confusion_matrix(true_actions, pred_actions) -> ConfusionMatrix:
    matrix = [[0 for _ in range(3)] for _ in range(3)]
    for t_row, p_row in zip(true_actions, pred_actions):
        for t, p in zip(t_row, p_row):
            matrix[int(t)][int(p)] += 1
    return ConfusionMatrix(matrix=matrix)


@dataclass
class SwitchMetrics:
    switch_point_recall: float = 0.0
    switch_direction_accuracy: float = 0.0
    switch_timing_error_mean: float = 0.0
    switch_timing_error_distribution: Dict[str, float] = field(default_factory=dict)


def _find_switch_point(actions: Sequence[int]) -> int:
    """返回 ``actions`` 中第一个切换的 index；没有切换返回 -1。

    DP demonstration 严格满足 single-trade，所以最多一个切换点。
    """
    for i in range(1, len(actions)):
        if actions[i] != actions[i - 1]:
            return i
    return -1


def switch_metrics(true_actions, pred_actions) -> SwitchMetrics:
    """切换点指标。

    实现要点
    --------
    - single-trade 假设: GT 至多 1 个切换点；prediction 可能 0 或多次切换。
      ``_find_switch_point`` 只取 *第一个* 切换点；多次切换会被
      ``single_trade_consistency_rate`` 单独标注。
    - GT 与 prediction 都全程 flat 时视为正确（``recall=1.0``）；
      该约定在设计 §9.5 与单测中均显式。
    - 方向准确率: GT 切换后的目标动作 vs prediction 切换后的目标动作。
    - timing error 单位: bar 数（绝对值）。
    """
    recall_hits = 0
    recall_total = 0
    direction_correct = 0
    direction_total = 0
    timing_errors: List[int] = []
    no_switch_correct = 0
    no_switch_total = 0
    for t_row, p_row in zip(true_actions, pred_actions):
        t_sw = _find_switch_point(t_row)
        p_sw = _find_switch_point(p_row)
        if t_sw == -1 and p_sw == -1:
            # 双方都不切换 → 视为 recall=1，并归入 no_switch
            no_switch_correct += 1
            no_switch_total += 1
            continue
        if t_sw == -1 and p_sw != -1:
            # 不该切换但切了 → 不计 recall，但方向错算 1 次错误
            direction_total += 1
            continue
        # GT 有切换
        recall_total += 1
        if p_sw != -1:
            recall_hits += 1
            timing_errors.append(abs(p_sw - t_sw))
            # 方向: t_row[t_sw] vs p_row[p_sw]
            direction_total += 1
            if t_row[t_sw] == p_row[p_sw]:
                direction_correct += 1

    total_recall_total = recall_total + no_switch_total
    total_recall_hits = recall_hits + no_switch_correct
    if total_recall_total == 0:
        recall = 0.0
    else:
        recall = total_recall_hits / total_recall_total
    direction_acc = direction_correct / max(direction_total, 1)
    if timing_errors:
        mean_err = sum(timing_errors) / len(timing_errors)
        distribution = {
            "mean": mean_err,
            "p50": _quantile(timing_errors, 0.5),
            "p90": _quantile(timing_errors, 0.9),
            "max": float(max(timing_errors)),
        }
    else:
        mean_err = 0.0
        distribution = {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return SwitchMetrics(
        switch_point_recall=recall,
        switch_direction_accuracy=direction_acc,
        switch_timing_error_mean=mean_err,
        switch_timing_error_distribution=distribution,
    )


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


def single_trade_consistency_rate(pred_actions) -> float:
    """``pred_actions`` 中切换次数 ≤ 1 的样本占比。"""
    total = 0
    consistent = 0
    for row in pred_actions:
        total += 1
        switches = sum(1 for i in range(1, len(row)) if row[i] != row[i - 1])
        if switches <= 1:
            consistent += 1
    return consistent / max(total, 1)
