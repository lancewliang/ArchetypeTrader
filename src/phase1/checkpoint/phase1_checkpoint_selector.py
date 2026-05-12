"""Phase I best-checkpoint selector skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence




class Phase1CheckpointSelector:
    """Select the best Phase I checkpoint from evaluator metrics."""

    def select_best(
        self,
        validation_checkpoints: List[Phase1ValidationCheckpoint],
    ):
        """Select the best checkpoint from explicit validation checkpoints."""

        ...

