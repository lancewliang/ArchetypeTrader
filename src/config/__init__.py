"""Training configuration entry points.

Phase I offline preprocessing config lives in ``src.preprocess_data.config``.
``src.config.phase1_config`` keeps model/training/evaluation config and
re-exports shared preprocessing config classes for compatibility.
"""

from .phase1_config import Phase1Config  # noqa: F401
