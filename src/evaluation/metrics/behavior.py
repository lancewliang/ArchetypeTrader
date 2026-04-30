"""Codebook / decoder 行为多样性指标.

设计文档锚点: §4.11 与 §6.5。
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Literal, Sequence


def per_code_action_entropy(decoded_logits_by_code) -> Dict[int, float]:
    """每个 code 解出 action 分布的熵。

    Parameters
    ----------
    decoded_logits_by_code : ``Dict[code_id, list[N, h, 3]]``。
        固定一批 states 后，分别用每个 code 跑 decoder 得到的 logits 集合。

    Returns
    -------
    ``Dict[code_id, entropy]``。低熵 → archetype 退化为单一动作（设计 §6.5 行为多样性）。
    """
    out: Dict[int, float] = {}
    for cid, logits in decoded_logits_by_code.items():
        # action 分布: argmax 后再频次统计
        counts = [0, 0, 0]
        total = 0
        for batch_row in logits:
            for step in batch_row:
                pred = int(_argmax_seq(step))
                counts[pred] += 1
                total += 1
        if total == 0:
            out[cid] = 0.0
            continue
        probs = [c / total for c in counts]
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p)
        out[cid] = entropy
    return out


def _argmax_seq(seq):
    best_i = 0
    best_v = seq[0]
    for i in range(1, len(seq)):
        if seq[i] > best_v:
            best_v = seq[i]
            best_i = i
    return best_i


def inter_code_action_diversity(
    decoded_actions_by_code: Dict[int, List[List[int]]],
    distance: Literal["hamming"] = "hamming",
) -> float:
    """固定 states 下，不同 code 输出 action 序列的平均距离。

    实现
    ----
    - 当前仅支持 ``hamming``；返回 ``mean(per-pair, per-batch hamming / h)``。
    - DTW 较贵，留作未来扩展。

    用途: 检测 codebook 向量距离很远但 decoder 对 ``z_q`` 不敏感导致输出几乎相同
    的退化情况（设计 §4.11 / §6.5 + selection_policy.behavior guardrail）。
    """
    code_ids = sorted(decoded_actions_by_code.keys())
    if len(code_ids) < 2:
        return 0.0
    total = 0.0
    pair_count = 0
    # 对每对 (i, j)
    for ai in range(len(code_ids)):
        for aj in range(ai + 1, len(code_ids)):
            ca = decoded_actions_by_code[code_ids[ai]]
            cb = decoded_actions_by_code[code_ids[aj]]
            if not ca or not cb:
                continue
            for ra, rb in zip(ca, cb):
                if not ra or len(ra) != len(rb):
                    continue
                diff = sum(1 for x, y in zip(ra, rb) if x != y)
                total += diff / len(ra)
                pair_count += 1
    if pair_count == 0:
        return 0.0
    return total / pair_count


def decoder_sensitivity_to_code(decoded_logits_by_code) -> float:
    """固定 states 仅替换 ``z_q``，logits 的平均变化幅度。

    实现
    ----
    - 对每对 code，逐元素求 ``|logits_a - logits_b|.mean()``，最后再平均。
    - 接近 0 → decoder 几乎忽略 ``z_q``；selection_policy 的 behavior guardrail
      会拦截这类 checkpoint。
    """
    code_ids = sorted(decoded_logits_by_code.keys())
    if len(code_ids) < 2:
        return 0.0
    total = 0.0
    pair_count = 0
    for ai in range(len(code_ids)):
        for aj in range(ai + 1, len(code_ids)):
            la = decoded_logits_by_code[code_ids[ai]]
            lb = decoded_logits_by_code[code_ids[aj]]
            for batch_a, batch_b in zip(la, lb):
                # batch_a/b: [h, 3]
                if len(batch_a) == 0:
                    continue
                step_count = 0
                step_sum = 0.0
                for sa, sb in zip(batch_a, batch_b):
                    for x, y in zip(sa, sb):
                        step_sum += abs(x - y)
                        step_count += 1
                if step_count > 0:
                    total += step_sum / step_count
                    pair_count += 1
    if pair_count == 0:
        return 0.0
    return total / pair_count


def inter_code_distance(codebook) -> float:
    """codebook 向量两两 L2 距离平均值。

    与 ``inter_code_action_diversity`` 互补:
    - distance 高 + diversity 低 → codebook 已分开但 decoder 不响应；
    - distance 低 → 真正的 codebook collapse；
    - 二者都高 → 健康。
    """
    n = len(codebook)
    if n < 2:
        return 0.0
    total = 0.0
    pair = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = 0.0
            for x, y in zip(codebook[i], codebook[j]):
                d += (x - y) ** 2
            total += math.sqrt(d)
            pair += 1
    return total / pair


def latent_silhouette_score(latents, code_ids) -> float:
    """简化版 silhouette score: 用 cluster 中心近似两两距离矩阵。

    实现
    ----
    - 对每个 ``z_e``: ``a = ||z_e - own_center||``，``b = min(||z_e - other_center||)``。
    - silhouette = ``(b - a) / max(a, b)``，对所有样本取均值。
    - 完整 silhouette 计算复杂度 O(N²)，对大 batch 太慢；中心近似在 K 较小时精度可用。
    """
    cluster_points: Dict[int, List] = defaultdict(list)
    for vec, cid in zip(latents, code_ids):
        cluster_points[int(cid)].append(vec)
    if len(cluster_points) < 2:
        return 0.0
    centers = {cid: _mean_vec(vecs) for cid, vecs in cluster_points.items()}
    scores: List[float] = []
    for vec, cid in zip(latents, code_ids):
        a = _l2(vec, centers[int(cid)])
        # 找到最近的非自身 cluster 中心
        b = min(_l2(vec, c) for k, c in centers.items() if k != int(cid))
        denom = max(a, b)
        if denom == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / denom)
    return sum(scores) / max(len(scores), 1)


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        s += (x - y) ** 2
    return math.sqrt(s)


def _mean_vec(vecs: List[Sequence[float]]) -> List[float]:
    if not vecs:
        return []
    dim = len(vecs[0])
    out = [0.0] * dim
    for v in vecs:
        for i in range(dim):
            out[i] += v[i]
    return [x / len(vecs) for x in out]
