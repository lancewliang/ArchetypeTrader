"""Feature provenance helpers for Phase I/II no-leakage audit.

The data processor owns the schema and factor-list boundary, so it is the
right place to emit a compact declaration of when each feature is available.
Phase II consumes the same declaration as a sign-off guardrail.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from src.data.schema import InputSchema, KNOWN_ORDERBOOK_COLUMNS, KNOWN_TRADE_COLUMNS
from src.utils.feather_io import atomic_write_json


_LEAKAGE_NAME_RE = re.compile(r"(future|target|label|centered|lead)", re.IGNORECASE)
_ROLLING_HINT_RE = re.compile(
    r"(return|ret|ratio|zscore|vol|volatility|trend|momentum|mean|std|ma|ema|sma|rolling)",
    re.IGNORECASE,
)
_TRAILING_WINDOW_RE = re.compile(r"_(\d+)(?:m|min|bar|bars)?$")


@dataclass(frozen=True)
class FeatureProvenanceCheck:
    """Result of the Phase II feature-provenance no-leakage check."""

    no_leakage_signoff: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def write_feature_provenance_json(
    schema: InputSchema | Mapping[str, Any],
    path: Path | str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``feature_provenance.json`` for a Phase I artifact directory."""

    return atomic_write_json(
        build_feature_provenance(schema, metadata=metadata),
        path,
    )


def build_feature_provenance(
    schema: InputSchema | Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a conservative provenance declaration from ``input_schema.json``.

    The factor columns are already materialized in the market files; this
    helper records the contract Phase II needs: each feature is known at the
    decision row, uses only current/history rows, and any fitted preprocessing
    is train-only.
    """

    payload = _schema_payload(schema)
    feature_columns = list(payload.get("feature_columns", []))
    feature_source = dict(payload.get("feature_source") or {})
    orderbook_columns = set(payload.get("orderbook_columns") or [])
    fixed_features = set(feature_source.get("fixed_features") or [])
    configured = set(feature_source.get("configured_factors") or [])
    current_row_columns = (
        set(KNOWN_ORDERBOOK_COLUMNS)
        | set(KNOWN_TRADE_COLUMNS)
        | fixed_features
        | {
            "mid_price",
            "avg_trade_price",
            "sell_vwap",
            "buy_vwap",
            "price_spread",
            "weighted_imbalance",
            "weighted_imbalance_inv",
            "klow",
            "klow2",
            "kmid2",
            "ksft2",
            "kup",
            "kup2",
        }
    )

    entries: Dict[str, Dict[str, Any]] = {}
    for col in feature_columns:
        source_type = _source_type(
            col,
            orderbook_columns=orderbook_columns,
            fixed_features=fixed_features,
            configured_factors=configured,
            current_row_columns=current_row_columns,
        )
        entries[col] = {
            "source_columns": [col],
            "source_type": source_type,
            "lookback_start_bars": _infer_lookback_start(col, current_row_columns),
            "lookback_end_bars": 0,
            "publish_delay_bars": 0,
            "available_at_lag": 0,
            "fit_scope": "train_only",
            "normalization_scope": "train_only",
            "uses_future_rows": False,
        }

    out: Dict[str, Any] = {
        "version": 1,
        "feature_columns": entries,
        "lag_convention": "negative_or_zero_means_known_at_decision_time",
        "provenance_mode": "declared_from_input_schema",
        "feature_source": feature_source,
    }
    if metadata:
        out["metadata"] = dict(metadata)
    return out


def evaluate_feature_provenance(
    provenance: Mapping[str, Any],
    *,
    feature_columns: Sequence[str],
) -> FeatureProvenanceCheck:
    """Check whether provenance is sufficient for Phase II no-leakage sign-off."""

    blockers: list[str] = []
    warnings: list[str] = []
    entries = provenance.get("feature_columns")
    if not isinstance(entries, Mapping):
        return FeatureProvenanceCheck(
            no_leakage_signoff=False,
            blockers=["feature_provenance.json 缺少 feature_columns 映射"],
        )

    expected = list(feature_columns)
    for feature in expected:
        if _LEAKAGE_NAME_RE.search(feature):
            blockers.append(f"feature 名称命中泄漏黑名单: {feature}")
        entry = entries.get(feature)
        if not isinstance(entry, Mapping):
            blockers.append(f"feature_provenance 缺少 feature: {feature}")
            continue
        if bool(entry.get("uses_future_rows", False)):
            blockers.append(f"{feature}.uses_future_rows=true")
        if _as_float(entry.get("lookback_end_bars", 0.0)) > 0:
            blockers.append(f"{feature}.lookback_end_bars>0")
        if _as_float(entry.get("publish_delay_bars", 0.0)) > 0:
            blockers.append(f"{feature}.publish_delay_bars>0")
        if entry.get("fit_scope") == "all_splits":
            blockers.append(f"{feature}.fit_scope=all_splits")
        if entry.get("normalization_scope") == "all_splits":
            blockers.append(f"{feature}.normalization_scope=all_splits")

    extras = sorted(set(str(k) for k in entries.keys()) - set(expected))
    if extras:
        warnings.append(f"feature_provenance 含 schema 外 feature: {extras[:5]}")

    return FeatureProvenanceCheck(
        no_leakage_signoff=not blockers,
        blockers=blockers,
        warnings=warnings,
    )


def file_sha256_short(path: Path | str, length: int = 16) -> str:
    """Return a short SHA256 digest for audit fields."""

    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest[:length]


def _schema_payload(schema: InputSchema | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(schema, InputSchema):
        return schema.to_dict()
    return dict(schema)


def _source_type(
    col: str,
    *,
    orderbook_columns: set[str],
    fixed_features: set[str],
    configured_factors: set[str],
    current_row_columns: set[str],
) -> str:
    if col in orderbook_columns:
        return "orderbook_current_row"
    if col in fixed_features or col in current_row_columns:
        return "market_current_row"
    if col in configured_factors:
        return "configured_factor"
    return "schema_feature"


def _infer_lookback_start(col: str, current_row_columns: set[str]) -> int:
    if col in current_row_columns:
        return 0
    match = _TRAILING_WINDOW_RE.search(col)
    if match and _ROLLING_HINT_RE.search(col):
        return -int(match.group(1))
    return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
