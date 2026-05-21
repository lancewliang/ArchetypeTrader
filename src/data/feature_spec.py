"""Feature block specifications for multi-input state tensors.

This module only describes how input columns are grouped. It does not load
market data, fit normalizers, or build tensors.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .resolve_factor import FACTORS_ROOT, FIXED_FEATURES, read_factor_config


PRICE_COLUMNS = ("close",)
FIXED_STATE_FEATURES = tuple(
    column for column in FIXED_FEATURES if column not in PRICE_COLUMNS
)


def _dedupe_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    """Return unique values while preserving first occurrence order."""

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class FeatureBlock:
    """A named group of feature columns with optional block-level normalization."""

    name: str
    columns: tuple[str, ...]
    normalize: bool = False
    normalizer_key: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("feature block name must not be empty")
        if not self.columns:
            raise ValueError(f"feature block {self.name!r} must contain columns")
        if any(not column for column in self.columns):
            raise ValueError(f"feature block {self.name!r} contains an empty column")

    @classmethod
    def from_columns(
        cls,
        *,
        name: str,
        columns: Sequence[str],
        normalize: bool,
        normalizer_key: str | None = None,
    ) -> FeatureBlock:
        """Create a block from an explicit column sequence."""

        return cls(
            name=name,
            columns=tuple(columns),
            normalize=normalize,
            normalizer_key=normalizer_key or name,
        )

    @classmethod
    def from_file(
        cls,
        *,
        name: str,
        path: str | Path,
        normalize: bool,
        normalizer_key: str | None = None,
    ) -> FeatureBlock:
        """Create a block from a factor config file."""

        return cls.from_columns(
            name=name,
            columns=read_factor_config(path),
            normalize=normalize,
            normalizer_key=normalizer_key,
        )

    @property
    def effective_normalizer_key(self) -> str:
        """Return the key used to save/load this block's normalizer."""

        return self.normalizer_key or self.name

    @property
    def output_columns(self) -> tuple[str, ...]:
        """Return block-qualified output feature names for metadata/debugging."""

        return tuple(f"{self.name}::{column}" for column in self.columns)


@dataclass(frozen=True)
class FeatureInputSpec:
    """Three-input feature spec consumed by horizon building."""

    state_blocks: tuple[FeatureBlock, ...]
    relative_state_blocks: tuple[FeatureBlock, ...]
    trend_state_blocks: tuple[FeatureBlock, ...]
    price_columns: tuple[str, ...] = PRICE_COLUMNS

    def __post_init__(self) -> None:
        if not self.state_blocks:
            raise ValueError("state_blocks must not be empty")
        if not self.relative_state_blocks:
            raise ValueError("relative_state_blocks must not be empty")
        if not self.trend_state_blocks:
            raise ValueError("trend_state_blocks must not be empty")
        if not self.price_columns:
            raise ValueError("price_columns must not be empty")

    def iter_blocks(self) -> Iterator[FeatureBlock]:
        """Iterate over all blocks in tensor-construction order."""

        yield from self.state_blocks
        yield from self.relative_state_blocks
        yield from self.trend_state_blocks

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Raw dataframe columns needed to build all tensors."""

        columns: list[str] = [*self.price_columns]
        for block in self.iter_blocks():
            columns.extend(block.columns)
        return _dedupe_preserve_order(columns)

    @property
    def state_columns(self) -> tuple[str, ...]:
        """Block-qualified columns emitted into the ``states`` tensor."""

        return tuple(
            column
            for block in self.state_blocks
            for column in block.output_columns
        )

    @property
    def relative_state_columns(self) -> tuple[str, ...]:
        """Block-qualified columns emitted into the ``relative_states`` tensor."""

        return tuple(
            column
            for block in self.relative_state_blocks
            for column in block.output_columns
        )

    @property
    def trend_state_columns(self) -> tuple[str, ...]:
        """Block-qualified columns emitted into the ``trend_states`` tensor."""

        return tuple(
            column
            for block in self.trend_state_blocks
            for column in block.output_columns
        )

    @property
    def normalizer_keys(self) -> tuple[str, ...]:
        """Normalizer keys required by normalized blocks."""

        return _dedupe_preserve_order(
            block.effective_normalizer_key
            for block in self.iter_blocks()
            if block.normalize
        )


def build_feature_input_spec(
    *,
    pair: str,
    factors_root: str | Path = FACTORS_ROOT,
) -> FeatureInputSpec:
    """Build the default three-input feature spec for a trading pair."""

    pair_root = Path(factors_root) / pair
    return FeatureInputSpec(
        state_blocks=(
            FeatureBlock.from_file(
                name="remaining_need_normalization",
                path=pair_root / "remaining_need_normalization.md",
                normalize=True,
            ),
            FeatureBlock.from_columns(
                name="fixed_state_features",
                columns=FIXED_STATE_FEATURES,
                normalize=True,
            ),
        ),
        relative_state_blocks=(
            FeatureBlock.from_file(
                name="relative_need_normalization",
                path=pair_root / "relative_need_normalization.md",
                normalize=True,
            ),
            FeatureBlock.from_file(
                name="relative",
                path=pair_root / "relative.md",
                normalize=False,
            ),
        ),
        trend_state_blocks=(
            FeatureBlock.from_file(
                name="trend_long",
                path=pair_root / "trend_long.md",
                normalize=False,
            ),
            FeatureBlock.from_file(
                name="trend_long_need_normalization",
                path=pair_root / "trend_long_need_normalization.md",
                normalize=True,
            ),
            FeatureBlock.from_file(
                name="trend_short",
                path=pair_root / "trend_short.md",
                normalize=False,
            ),
            FeatureBlock.from_file(
                name="trend_short_need_normalization",
                path=pair_root / "trend_short_need_normalization.md",
                normalize=True,
            ),
        ),
    )


__all__ = [
    "FeatureBlock",
    "FeatureInputSpec",
    "FIXED_STATE_FEATURES",
    "PRICE_COLUMNS",
    "build_feature_input_spec",
]
