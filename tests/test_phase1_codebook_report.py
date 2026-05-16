from __future__ import annotations

from src.phase1.report.phase1_codebook_report import Phase1CodebookReport
from src.phase1.report.phase1_codebook_report_context import (
    Phase1CodebookReportContextBuilder,
)


class _ValidationResult:
    checkpoint_id = "vq_epoch_0007"
    stage = "vq"
    epoch = 7
    passed = True
    score = 0.9
    failed_layers = ()
    layers = ()
    code_diagnostics = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "stage": self.stage,
            "epoch": self.epoch,
            "passed": self.passed,
            "score": self.score,
            "failed_layers": [],
            "layers": [],
            "code_diagnostics": [],
            "drift_diagnostics": {},
            "tie_breaker_metrics": {},
        }


def _metric_payload(
    *,
    name: str = "validation_action_accuracy",
    value: float = 0.91234567,
    threshold: str = ">= 0.85",
    severity: str = "pass",
    passed: bool = True,
    message: str = "ok",
) -> dict[str, object]:
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "severity": severity,
        "passed": passed,
        "layer": "vq_internal",
        "message": message,
    }


def _payload() -> dict[str, object]:
    return {
        "report": {
            "title": "Custom <Title>",
            "generated_at": "2026-05-10T00:00:00+00:00",
            "schema": "phase1_codebook_validation_report.v1",
        },
        "summary": {
            "checkpoint_id": "ckpt<1>",
            "stage": "validation",
            "epoch": 7,
            "passed": False,
            "score": 0.87654321,
            "failed_layers": ["vq<internal>"],
            "code_diagnostic_count": 1,
        },
        "validation": {
            "layers": [
                {
                    "layer_id": 1,
                    "name": "vq<internal>",
                    "passed": False,
                    "metrics": [
                        _metric_payload(message="needs <escape>"),
                        _metric_payload(
                            name="dead_code_ratio",
                            value=0.3,
                            threshold="<= 0.1",
                            severity="fail",
                            passed=False,
                            message="too high",
                        ),
                    ],
                }
            ],
            "code_diagnostics": [
                {
                    "code_id": 3,
                    "support": 24,
                    "occupancy": 0.125,
                    "dominant_morphology": "trend<up>",
                    "dominant_morphology_ratio": 0.67,
                    "morphology_lift": None,
                    "dominant_motif": "hold",
                    "dominant_motif_ratio": 0.5,
                    "dominant_pair": "trend-hold",
                    "dominant_pair_ratio": 0.45,
                    "decoded_mean_advantage": 1.25,
                    "decoded_win_rate": 0.55,
                    "retention_ratio": None,
                    "fee_drag": 0.02,
                    "status": "weak<script>",
                }
            ],
            "drift_diagnostics": {
                "assignment_churn": _metric_payload(
                    name="assignment_churn",
                    value=float("nan"),
                    threshold="< 0.2",
                    severity="warn",
                    message="watch",
                )
            },
            "tie_breaker_metrics": {
                "risk_adjusted_return": 1.23456789,
                "active_code_ratio": True,
            },
            "score": {
                "total_score": 0.87654321,
                "components": [
                    {
                        "name": "teacher_quality",
                        "value": 0.5,
                        "weight": 0.1,
                        "weighted_value": 0.05,
                    }
                ],
            },
            "vq_internal_payload": {
                "code_distribution": [0.5, 0.25, 0.0],
                "active_codes": [0, 1],
                "current_assignment": {
                    "epoch": 7,
                    "split": "val",
                    "sample_ids": [0, 1, 2],
                    "code_ids": [0, 1, 0],
                    "active_codes": [0, 1],
                },
                "assignment_churn_by_epoch": {},
                "codebook_size": 3,
                "codebook_size_available": True,
                "code_distribution_sample_count": 3,
            },
        },
        "config": {"codebook_size": 32},
        "artifacts": {"checkpoint": "/tmp/ckpt.pt"},
    }


def test_render_html_uses_template_and_escapes_values() -> None:
    html = Phase1CodebookReport().render_html(_payload())

    assert "<!doctype html>" in html
    assert "<title>Custom &lt;Title&gt;</title>" in html
    assert "ckpt&lt;1&gt;" in html
    assert "vq&lt;internal&gt;" in html
    assert "needs &lt;escape&gt;" in html
    assert "weak&lt;script&gt;" in html
    assert "0.876543" in html
    assert "NaN" in html
    assert "risk_adjusted_return" in html
    assert "{%" not in html
    assert "{{" not in html


def test_build_html_builds_payload_and_renders_template() -> None:
    html = Phase1CodebookReport().build_html(
        validation_result=_ValidationResult(),
        config={"codebook_size": 32},
        artifacts={"checkpoint": "/tmp/ckpt.pt"},
    )

    assert "<!doctype html>" in html
    assert "vq_epoch_0007" in html
    assert "codebook_size" in html
    assert "/tmp/ckpt.pt" in html


def test_empty_optional_sections_are_omitted() -> None:
    payload = _payload()
    validation = payload["validation"]
    assert isinstance(validation, dict)
    validation["code_diagnostics"] = []
    validation["drift_diagnostics"] = {}
    validation["tie_breaker_metrics"] = {}
    payload["config"] = {}
    payload["artifacts"] = {}

    html = Phase1CodebookReport().render_html(payload)

    assert "Code Diagnostics</h2>" not in html
    assert "Drift Diagnostics</h2>" not in html
    assert "Tie Breaker Metrics</h2>" not in html
    assert "Config Snapshot</h2>" not in html
    assert "Artifacts</h2>" not in html


def test_build_html_context_includes_code_distribution_rows() -> None:
    context = Phase1CodebookReportContextBuilder().build(_payload())

    assert context["code_distribution"] == [
        {
            "code_id": "0",
            "occupancy": "0.5",
            "occupancy_percent": "50%",
            "bar_width": "50%",
            "active": True,
            "badge_class": "pass",
            "status_label": "ACTIVE",
        },
        {
            "code_id": "1",
            "occupancy": "0.25",
            "occupancy_percent": "25%",
            "bar_width": "25%",
            "active": True,
            "badge_class": "pass",
            "status_label": "ACTIVE",
        },
        {
            "code_id": "2",
            "occupancy": "0",
            "occupancy_percent": "0%",
            "bar_width": "0%",
            "active": False,
            "badge_class": "warn",
            "status_label": "INACTIVE",
        },
    ]


def test_build_html_context_includes_score_breakdown_rows() -> None:
    context = Phase1CodebookReportContextBuilder().build(_payload())

    assert context["score_breakdown_rows"] == [
        {
            "name": "teacher_quality",
            "value": "0.5",
            "weight": "0.1",
            "weighted_value": "0.05",
        }
    ]
