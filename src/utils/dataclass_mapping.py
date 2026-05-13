"""Dataclass serialization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any, TypeVar


TDataclass = TypeVar("TDataclass")


def _dataclass_from_mapping(
    config_type: type[TDataclass],
    payload: Mapping[str, Any],
) -> TDataclass:
    """Build a dataclass from a mapping, ignoring unknown keys."""

    field_names = {field.name for field in fields(config_type)}
    values = {key: value for key, value in payload.items() if key in field_names}
    return config_type(**values)


__all__ = ["_dataclass_from_mapping"]
