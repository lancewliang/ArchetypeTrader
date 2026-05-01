"""Phase II dead-code mask helpers.

Phase II selector should inherit Phase I global usage diagnostics and mask
archetypes whose usage ratio is below the configured threshold.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _extract_usage_values(phase1_report: Mapping[str, Any]) -> Sequence[float] | None:
    """Extract per-code usage counts or ratios from known Phase I report shapes."""
    for key in ("code_usage_ratio", "per_code_usage_ratio", "usage_ratio"):
        value = phase1_report.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [float(v) for v in value]

    code_usage = phase1_report.get("code_usage")
    if isinstance(code_usage, Mapping):
        for key in ("ratios", "ratio", "usage_ratio"):
            value = code_usage.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [float(v) for v in value]
        counts = code_usage.get("counts")
        if isinstance(counts, Sequence) and not isinstance(counts, (str, bytes)):
            return [float(v) for v in counts]

    counts = phase1_report.get("code_usage_counts")
    if isinstance(counts, Sequence) and not isinstance(counts, (str, bytes)):
        return [float(v) for v in counts]
    return None


def build_dead_code_mask(
    phase1_report: Mapping[str, Any],
    num_codes: int,
    threshold: float,
) -> list[bool]:
    """Build a dead-code mask from Phase I usage diagnostics.

    If usage data is absent, returns all False so inference/backtest can still
    run. Training code may choose to fail-fast before calling this helper.
    """
    values = _extract_usage_values(phase1_report)
    if values is None:
        return [False] * max(num_codes, 0)

    vals = [float(v) for v in values[:num_codes]]
    if len(vals) < num_codes:
        vals.extend([0.0] * (num_codes - len(vals)))

    total = sum(vals)
    if total > 1.0 + 1e-6:
        ratios = [v / total for v in vals]
    else:
        ratios = vals
    return [r < float(threshold) for r in ratios]

