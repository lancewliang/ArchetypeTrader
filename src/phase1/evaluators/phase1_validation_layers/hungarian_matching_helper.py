"""Hungarian matching helpers for VQ code id alignment.

本模块只处理跨 epoch code id 对齐，不计算 Layer 1 业务指标。调用方提供
``CodeAssignmentSnapshot`` 中保存的 code/action prototypes，本模块返回历史 code
id 到当前 code id 空间的映射。

算法目标:
    VQ code id 是可置换的离散编号，同一语义的 archetype 可能在相邻 epoch 中从
    code 3 变成 code 7。直接比较 raw code id 会把这种纯编号交换误判为 churn。
    因此这里先用 code 原型做 bipartite matching，把 previous epoch 的 code id
    投影到 current epoch 的 code id 空间，再交给 churn/lifetime 指标使用。

算法流程:
    1. 按 ``prototype_kind`` 选择原型矩阵，默认 ``auto`` 表示优先使用 decoded
       action prototype，缺失时退回 latent/code embedding prototype。
    2. 过滤掉包含 NaN/Inf 的 code 行。inactive/dead code 没有样本支持时通常被
       写成 NaN 行，不参与 matching。
    3. 构造 cost matrix，行是 previous valid code，列是 current valid code，
       ``cost[i, j] = ||prototype_prev_i - prototype_curr_j||_2``。
    4. 用 Hungarian algorithm 求解最小总代价的一对一匹配，得到
       ``previous_code -> current_code`` 映射。
    5. 如果两种 prototype 都不可用或 shape 不兼容，返回空映射；调用方会用
       ``alignment.get(old_id, old_id)`` 回退到 raw id 行为，保持旧历史兼容。
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except ImportError:  # pragma: no cover - exercised only when optional scipy is absent.
    _scipy_linear_sum_assignment = None

from ...metrics import CodeAssignmentSnapshot


def _prototype_preference(prototype_kind: str) -> tuple[str, ...]:
    """返回 code id 对齐时的 prototype 优先级。

    ``action`` 使用 decoded action/position path 原型，更贴近 archetype 的交易
    行为语义；``code`` 使用 VQ embedding 原型，更贴近 latent 空间几何结构。
    ``auto`` 与 ``action`` 的优先级一致，区别只在调用语义上表达“自动选择可用
    原型”。
    """

    if prototype_kind == "action":
        return ("action", "code")
    if prototype_kind == "code":
        return ("code", "action")
    if prototype_kind == "auto":
        return ("action", "code")
    raise ValueError("prototype_kind must be one of auto, action, code")


def _snapshot_prototypes(
    snapshot: CodeAssignmentSnapshot,
    kind: str,
) -> np.ndarray | None:
    """读取指定类型的 assignment prototype 矩阵。

    返回值 shape 约定:
        - ``action``: ``[K, H]``，每行是该 code 的 decoded position path 均值；
        - ``code``: ``[K, D]``，每行是该 code 的 latent/code embedding 均值。

    行号必须等于 raw code id，这样 Hungarian 返回的矩阵行列 index 才能再映射回
    原始 code id。
    """

    if kind == "action":
        return snapshot.action_prototypes
    if kind == "code":
        return snapshot.code_prototypes
    raise ValueError("kind must be action or code")


def _valid_prototype_rows(prototypes: np.ndarray) -> np.ndarray:
    """返回 prototype 矩阵中可用于 matching 的 code index。

    inactive/dead code 通常没有样本可聚合，调用方会把该行填为 NaN。Hungarian
    matching 只在两边都有有限 prototype 的 code 上求解；未参与匹配的 code 由
    上层保持 raw id fallback。
    """

    values = np.asarray(prototypes, dtype=np.float64)
    if values.ndim != 2:
        return np.asarray([], dtype=np.int64)
    return np.flatnonzero(np.all(np.isfinite(values), axis=1))


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """求解线性分配；优先使用 SciPy Hungarian，缺失时小矩阵精确回退。

    输入:
        ``cost`` 是二维代价矩阵，shape 为 ``[num_previous_codes,
        num_current_codes]``。矩阵可以不是方阵；Hungarian 会匹配较小一侧的全部
        元素，并保证总 cost 最小。

    输出:
        ``(rows, cols)``，表示 ``rows[t]`` 这个 previous-side 行匹配到
        ``cols[t]`` 这个 current-side 列。

    fallback:
        SciPy 不可用时，K <= 8 使用全排列枚举得到精确最优解；更大矩阵用贪心
        最近邻兜底。正常环境已在 requirements 中声明 scipy，因此大矩阵贪心路径
        只用于极简诊断环境，不作为生产主路径。
    """

    if _scipy_linear_sum_assignment is not None:
        rows, cols = _scipy_linear_sum_assignment(cost)
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)

    # SciPy 是 requirements 的显式依赖；这个分支只为极简环境保底。
    n_rows, n_cols = cost.shape
    if min(n_rows, n_cols) <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    if max(n_rows, n_cols) <= 8:
        if n_rows <= n_cols:
            best_cols: tuple[int, ...] | None = None
            best_cost = float("inf")
            for cols in permutations(range(n_cols), n_rows):
                total = float(sum(cost[row, col] for row, col in enumerate(cols)))
                if total < best_cost:
                    best_cost = total
                    best_cols = cols
            return (
                np.arange(n_rows, dtype=np.int64),
                np.asarray(best_cols or (), dtype=np.int64),
            )
        best_rows: tuple[int, ...] | None = None
        best_cost = float("inf")
        for rows in permutations(range(n_rows), n_cols):
            total = float(sum(cost[row, col] for col, row in enumerate(rows)))
            if total < best_cost:
                best_cost = total
                best_rows = rows
        return (
            np.asarray(best_rows or (), dtype=np.int64),
            np.arange(n_cols, dtype=np.int64),
        )

    remaining_rows = set(range(n_rows))
    remaining_cols = set(range(n_cols))
    matched_rows: list[int] = []
    matched_cols: list[int] = []
    while remaining_rows and remaining_cols:
        row, col = min(
            (
                (row, col)
                for row in remaining_rows
                for col in remaining_cols
            ),
            key=lambda item: cost[item[0], item[1]],
        )
        remaining_rows.remove(row)
        remaining_cols.remove(col)
        matched_rows.append(row)
        matched_cols.append(col)
    return (
        np.asarray(matched_rows, dtype=np.int64),
        np.asarray(matched_cols, dtype=np.int64),
    )


def align_previous_codes_to_current(
    previous: CodeAssignmentSnapshot,
    current: CodeAssignmentSnapshot,
    *,
    prototype_kind: str = "auto",
) -> dict[int, int]:
    """用 prototype matching 将历史 code id 投影到当前 code id 空间。

    参数:
        previous: 历史 epoch 的 assignment snapshot。其 prototype 行号是历史
            raw code id。
        current: 当前 epoch 的 assignment snapshot。其 prototype 行号是当前
            raw code id。
        prototype_kind: ``auto``、``action`` 或 ``code``。决定先尝试哪一种原型。

    返回:
        ``dict[previous_code_id, current_code_id]``。例如返回 ``{3: 7}`` 表示
        previous epoch 中的 code 3 与 current epoch 中的 code 7 原型最接近，应
        在 churn/lifetime 计算前把历史 code 3 视为当前 code 7。

    匹配细节:
        对每种候选 prototype，先检查两边矩阵都存在、都是二维、且 feature 维度
        一致。随后取两边有限行作为有效 code，计算所有 previous/current pair 的
        L2 距离矩阵，并用 Hungarian algorithm 找到最小总代价的一对一匹配。
        一旦某种 prototype 成功匹配，就返回该匹配，不再尝试后续类型。

    缺失策略:
        若 prototype 缺失、全是 NaN、shape 不兼容或 cost 非有限，则尝试下一个
        prototype 类型；所有类型都失败时返回空 dict。空 dict 是有意设计的兼容
        信号，调用方会自然退回 raw code id 比较。
    """

    for kind in _prototype_preference(prototype_kind):
        previous_prototypes = _snapshot_prototypes(previous, kind)
        current_prototypes = _snapshot_prototypes(current, kind)
        if previous_prototypes is None or current_prototypes is None:
            continue
        left = np.asarray(previous_prototypes, dtype=np.float64)
        right = np.asarray(current_prototypes, dtype=np.float64)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
            continue
        left_codes = _valid_prototype_rows(left)
        right_codes = _valid_prototype_rows(right)
        if left_codes.size == 0 or right_codes.size == 0:
            continue
        # cost[row, col] is the L2 distance between one historical code prototype
        # and one current code prototype.
        deltas = left[left_codes, None, :] - right[None, right_codes, :]
        cost = np.linalg.norm(deltas, axis=-1)
        if not np.all(np.isfinite(cost)):
            continue
        row_indices, col_indices = _linear_sum_assignment(cost)
        return {
            int(left_codes[row]): int(right_codes[col])
            for row, col in zip(row_indices, col_indices, strict=False)
        }
    return {}


def active_codes_aligned_to_current(
    previous: CodeAssignmentSnapshot,
    current: CodeAssignmentSnapshot,
    *,
    prototype_kind: str = "auto",
) -> set[int]:
    """把历史 active code 集合投影到当前 code id 空间。

    用途:
        lifetime 指标需要回答“当前 active code 的语义是否已连续活跃多个 epoch”。
        如果某个语义从历史 code 3 置换为当前 code 7，直接检查 ``7 in
        previous.active_codes`` 会失败。这里先调用
        ``align_previous_codes_to_current()``，再把历史 active code id 映射为当前
        code id。

    返回:
        一个 current-id-space 的 active code 集合。未匹配到的历史 active code
        保持原 raw id，用于兼容没有 prototype 的旧 assignment history。
    """

    alignment = align_previous_codes_to_current(
        previous,
        current,
        prototype_kind=prototype_kind,
    )
    return {
        alignment.get(int(code_id), int(code_id))
        for code_id in previous.active_codes
    }


__all__ = [
    "active_codes_aligned_to_current",
    "align_previous_codes_to_current",
]
