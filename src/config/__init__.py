"""Training configuration entry points.

Phase I offline preprocessing config lives in ``src.preprocess_data.config``.
``src.phase1.config`` keeps model/training/evaluation config and
re-exports shared preprocessing config classes for Phase I training.
"""

from src.phase1.config import Phase1Config  # noqa: F401
