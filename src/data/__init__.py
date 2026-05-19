"""Data processing utilities."""

from .data_load import DataLoad
from .state_normalizer import StateNormalizer
from ..model.data_types import (
    ArtifactPaths,
    DemonstrationTrajectory,
    HorizonDataset,
    TrajectoryDataset,
)
from .horizon_builder import HorizonBuilder

__all__ = [
    "ArtifactPaths",
    "DataLoad",
    "DataPreparer",
    "DataStore",
    "DemonstrationTrajectory",
    "HorizonBuilder",
    "HorizonDataset",
    "StateNormalizer",
    "TrajectoryDataset",
]


def __getattr__(name: str):
    if name == "DataPreparer":
        from .data_preparer import DataPreparer as exported

        return exported
    if name == "DataStore":
        from ..store.artifact_store import DataStore as exported

        return exported
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
