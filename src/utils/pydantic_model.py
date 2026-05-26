"""Project-wide Pydantic model defaults."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict


class PydanticBaseModel(BaseModel):
    """Base class for serializable project schemas."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        coerce_numbers_to_str=True,
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


class PydanticMappingModel(PydanticBaseModel, Mapping[str, object]):
    """Pydantic model with dict-like read access for legacy payloads."""

    def _mapping(self) -> dict[str, object]:
        """Return a Python-value mapping view."""

        return self.model_dump(mode="python")

    def __getitem__(self, key: str) -> object:
        """Read a field by its payload key."""

        return self._mapping()[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate payload keys."""

        return iter(self._mapping())

    def __len__(self) -> int:
        """Return the number of payload keys."""

        return len(self._mapping())


__all__ = ["PydanticBaseModel", "PydanticMappingModel"]
