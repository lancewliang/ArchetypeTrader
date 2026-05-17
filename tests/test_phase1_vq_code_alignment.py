from __future__ import annotations

import numpy as np

from src.phase1.evaluators.phase1_validation_layers.layer1_vq_internal import (
    compute_assignment_churn,
    compute_code_lifetime_pass_ratio,
)
from src.phase1.metrics import CodeAssignmentSnapshot


def _snapshot(
    *,
    epoch: int,
    code_ids: list[int],
    active_codes: tuple[int, ...],
    action_prototypes: list[list[float | None]] | None = None,
) -> CodeAssignmentSnapshot:
    prototypes = None
    if action_prototypes is not None:
        prototypes = np.asarray(
            [
                [np.nan if value is None else value for value in row]
                for row in action_prototypes
            ],
            dtype=np.float64,
        )
    return CodeAssignmentSnapshot(
        epoch=epoch,
        split="val",
        sample_ids=np.arange(len(code_ids), dtype=np.int64),
        code_ids=np.asarray(code_ids, dtype=np.int64),
        active_codes=active_codes,
        action_prototypes=prototypes,
    )


def test_assignment_churn_aligns_permuted_code_ids_by_action_prototype() -> None:
    previous = _snapshot(
        epoch=1,
        code_ids=[0, 0, 1, 1],
        active_codes=(0, 1),
        action_prototypes=[
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
    )
    current = _snapshot(
        epoch=2,
        code_ids=[1, 1, 0, 0],
        active_codes=(0, 1),
        action_prototypes=[
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
    )

    churn = compute_assignment_churn(current, [previous], 1)

    assert churn == 0.0


def test_assignment_churn_falls_back_to_raw_ids_without_prototypes() -> None:
    previous = _snapshot(
        epoch=1,
        code_ids=[0, 0, 1, 1],
        active_codes=(0, 1),
    )
    current = _snapshot(
        epoch=2,
        code_ids=[1, 1, 0, 0],
        active_codes=(0, 1),
    )

    churn = compute_assignment_churn(current, [previous], 1)

    assert churn == 1.0


def test_code_lifetime_aligns_historical_active_code_to_current_id() -> None:
    history = [
        _snapshot(
            epoch=epoch,
            code_ids=[1, 1, 1],
            active_codes=(1,),
            action_prototypes=[
                [None, None, None],
                [1.0, 1.0, 1.0],
            ],
        )
        for epoch in range(1, 5)
    ]
    current = _snapshot(
        epoch=5,
        code_ids=[0, 0, 0],
        active_codes=(0,),
        action_prototypes=[
            [1.0, 1.0, 1.0],
            [None, None, None],
        ],
    )

    pass_ratio = compute_code_lifetime_pass_ratio(
        current.active_codes,
        history,
        5,
        current_epoch=current.epoch,
        split=current.split,
        current_assignment=current,
    )

    assert pass_ratio == 1.0
