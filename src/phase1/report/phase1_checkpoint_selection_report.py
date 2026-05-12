"""Phase I checkpoint selection report renderer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
from pathlib import Path
from typing import Any, Mapping

from ._template import render_template_file
from ..checkpoint import Phase1CheckpointSelectionResult


JsonObject = dict[str, Any]
_TEMPLATE_PATH = Path(__file__).with_name("templates") / (
    "phase1_checkpoint_selection_report.html"
)


def _json_safe(value: Any) -> Any:
    """Convert common payload values into JSON/template friendly values."""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return value


def _format_value(value: Any) -> str:
    """Format values for compact HTML table display."""

    if value is None:
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, tuple | list):
        return ", ".join(str(item) for item in value) or "-"
    return str(value)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Best-effort conversion of dataclass-like report inputs to mappings."""

    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        return payload if isinstance(payload, Mapping) else {}
    if isinstance(value, Mapping):
        return value
    return {}


@dataclass(frozen=True)
class Phase1CheckpointSelectionReport:
    """Render a static HTML report from ``Phase1CheckpointSelectionResult``."""

    title: str = "第一阶段检验点选择报告"

    def build_payload(
        self,
        *,
        selection_result: Phase1CheckpointSelectionResult,
        config: Mapping[str, object] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> JsonObject:
        """Build the machine-readable payload shared by JSON and HTML views."""

        generated_at = datetime.now(UTC).isoformat()
        selection_payload = selection_result.to_dict()
        selected_payload = self._build_selected_payload(selection_result)
        rejected = selection_payload.get("rejected", ())
        summary = {
            "has_selection": selection_result.has_selection,
            "selected_checkpoint_id": selection_result.selected_checkpoint_id,
            "selected_epoch": selection_result.selected_epoch,
            "selected_score": selection_result.selected_score,
            "candidate_count": selection_result.candidate_count,
            "eligible_count": selection_result.eligible_count,
            "rejected_count": len(rejected),
            "reason": selection_result.reason,
        }
        payload = {
            "report": {
                "title": self.title,
                "generated_at": generated_at,
                "schema": "phase1_checkpoint_selection_report.v1",
                **dict(metadata or {}),
            },
            "summary": summary,
            "selection": selection_payload,
            "selected": selected_payload,
            "config": dict(config or {}),
            "artifacts": dict(artifacts or {}),
        }
        return _json_safe(payload)

    def build_html(
        self,
        *,
        selection_result: Phase1CheckpointSelectionResult,
        config: Mapping[str, object] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> str:
        """Build a static HTML report from a checkpoint selection result."""

        payload = self.build_payload(
            selection_result=selection_result,
            config=config,
            artifacts=artifacts,
            metadata=metadata,
        )
        return self.render_html(payload)

    def render_html(self, payload: Mapping[str, Any]) -> str:
        """Render a previously built payload into an HTML string."""

        return render_template_file(_TEMPLATE_PATH, self._build_html_context(payload))

    def _build_selected_payload(
        self,
        selection_result: Phase1CheckpointSelectionResult,
    ) -> JsonObject:
        """Extract selected checkpoint details without duplicating model payloads."""

        selected = selection_result.selected
        if selected is None:
            return {}

        validation = getattr(selected, "codebook_validation", None)
        return {
            "stage": getattr(selected, "stage", None),
            "epoch": getattr(selected, "epoch", None),
            "train": _as_mapping(getattr(selected, "train", None)),
            "val": _as_mapping(getattr(selected, "val", None)),
            "codebook_validation": {
                "checkpoint_id": getattr(validation, "checkpoint_id", None),
                "passed": getattr(validation, "passed", None),
                "score": getattr(validation, "score", None),
                "failed_layers": tuple(getattr(validation, "failed_layers", ())),
                "tie_breaker_metrics": _as_mapping(
                    getattr(validation, "tie_breaker_metrics", None)
                ),
            },
        }

    def _build_html_context(self, payload: Mapping[str, Any]) -> JsonObject:
        """Convert report payload into the template display model."""

        report = _as_mapping(payload.get("report", {}))
        summary = _as_mapping(payload.get("summary", {}))
        selection = _as_mapping(payload.get("selection", {}))
        selected = _as_mapping(payload.get("selected", {}))
        validation = _as_mapping(selected.get("codebook_validation"))
        has_selection = bool(summary.get("has_selection", False))
        return {
            "page_title": str(report.get("title", self.title)),
            "header_title": self.title,
            "report": {
                "generated_at": str(report.get("generated_at", "-")),
                "schema": str(report.get("schema", "-")),
            },
            "summary": {
                "status_label": "已选中" if has_selection else "已阻断",
                "badge_class": "pass" if has_selection else "fail",
                "reason": str(summary.get("reason", "-")),
                "selected_checkpoint_id": _format_value(
                    summary.get("selected_checkpoint_id")
                ),
                "selected_epoch": _format_value(summary.get("selected_epoch")),
                "selected_score": _format_value(summary.get("selected_score")),
                "candidate_count": _format_value(summary.get("candidate_count")),
                "eligible_count": _format_value(summary.get("eligible_count")),
                "rejected_count": _format_value(summary.get("rejected_count")),
            },
            "selected": {
                "exists": has_selection,
                "stage": _format_value(selected.get("stage")),
                "epoch": _format_value(selected.get("epoch")),
                "checkpoint_id": _format_value(validation.get("checkpoint_id")),
                "passed": _format_value(validation.get("passed")),
                "score": _format_value(validation.get("score")),
                "failed_layers": _format_value(validation.get("failed_layers", ())),
            },
            "train_metric_rows": self._build_mapping_rows(
                _as_mapping(selected.get("train"))
            ),
            "val_metric_rows": self._build_mapping_rows(
                _as_mapping(selected.get("val"))
            ),
            "tie_breaker_rows": self._build_mapping_rows(
                _as_mapping(validation.get("tie_breaker_metrics"))
            ),
            "rejected_rows": [
                self._build_rejected_row(item)
                for item in selection.get("rejected", ())
            ],
            "config_rows": self._build_mapping_rows(
                _as_mapping(payload.get("config", {}))
            ),
            "artifact_rows": self._build_mapping_rows(
                _as_mapping(payload.get("artifacts", {}))
            ),
        }

    def _build_rejected_row(self, payload: Mapping[str, Any]) -> JsonObject:
        """Build a single rejected checkpoint row for the template."""

        passed = bool(payload.get("passed", False))
        return {
            "checkpoint_id": _format_value(payload.get("checkpoint_id")),
            "stage": _format_value(payload.get("stage")),
            "epoch": _format_value(payload.get("epoch")),
            "passed": _format_value(payload.get("passed")),
            "score": _format_value(payload.get("score")),
            "failed_layers": _format_value(payload.get("failed_layers", ())),
            "reason": _format_value(payload.get("reason")),
            "badge_class": "pass" if passed else "fail",
        }

    def _build_mapping_rows(self, payload: Mapping[str, Any]) -> list[JsonObject]:
        """Build key/value table rows."""

        return [
            {"key": str(key), "value": _format_value(value)}
            for key, value in payload.items()
        ]


__all__ = ["Phase1CheckpointSelectionReport"]
