"""Project-wide Pydantic model defaults."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict


class PydanticBaseModel(BaseModel):
    """Base class for serializable project schemas."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
        frozen=True,
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""

        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Restore a model from a mapping payload."""

        return cls.model_validate(payload)


__all__ = ["PydanticBaseModel"]
