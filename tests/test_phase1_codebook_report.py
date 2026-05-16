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
    risk_findings = ()

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
            "metrics": {
                "oracle_profitability": {
                    "mean_decoded_advantage_vs_flat": 0.174,
                    "random_label_relative_lift": 0.28,
                    "top_5_contribution": 0.42,
                    "trimmed_decoded_advantage": 0.091,
                },
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
            "oracle_profitability_payload": {
                "per_code_profitability": [],
                "decoded_returns": [0.2, -0.05, 0.1],
                "dp_returns": [0.3, -0.1, 0.2],
                "flat_returns": [0.0, 0.0, 0.0],
                "random_label_returns": [0.05, -0.02, 0.01],
                "random_seed": 7,
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


def test_build_html_context_includes_header_metadata() -> None:
    payload = _payload()
    report = payload["report"]
    assert isinstance(report, dict)
    report.update({"pair": "BTCUSDT", "batchid": "phase1_2026w18"})
    payload["config"] = {
        "pair": "ETHUSDT",
        "batch": "config_batch",
        "train_batch_id": "config_batch",
        "horizon": 72,
        "num_archetypes": 99,
    }

    context = Phase1CodebookReportContextBuilder().build(payload)

    assert context["header"] == {
        "pair": "BTCUSDT",
        "batch": "phase1_2026w18",
        "checkpoint": "ckpt<1>",
        "k": "3",
        "n_val": "3",
        "horizon": "72",
        "generated_at": "2026-05-10T00:00:00+00:00",
        "meta_items": [
            {"label": "Pair", "value": "BTCUSDT"},
            {"label": "Batch", "value": "phase1_2026w18"},
            {"label": "Checkpoint", "value": "ckpt<1>"},
            {"label": "K", "value": "3"},
            {"label": "N_val", "value": "3"},
            {"label": "Horizon", "value": "72"},
        ],
    }


def test_build_html_context_includes_three_part_risk_summary() -> None:
    payload = _payload()
    validation = payload["validation"]
    assert isinstance(validation, dict)
    validation["risk_findings"] = [
        {
            "severity": "warn",
            "title": "Drift warning",
            "reason": "motif drift",
            "related_metrics": ["motif_distribution_kl"],
            "related_codes": [],
            "related_pairs": [],
            "recommended_action": "compare train and validation motif distribution",
        },
        {
            "severity": "fail",
            "title": "Code 8 risk",
            "reason": "decoded return is negative",
            "related_metrics": ["decoded_mean_advantage"],
            "related_codes": [8],
            "related_pairs": ["neutral:mixed + middle + switching"],
            "recommended_action": "inspect code 8 decoded actions",
        },
    ]

    context = Phase1CodebookReportContextBuilder().build(payload)

    assert context["risk_summary"] == {
        "has_findings": True,
        "severity": "fail",
        "badge_class": "fail",
        "finding_count": "2",
        "primary_risk": "Code 8 risk: decoded return is negative",
        "inspection_target": (
            "优先检查 codes 8；pairs neutral:mixed + middle + switching；"
            "metrics decoded_mean_advantage。"
        ),
        "recommendation": "inspect code 8 decoded actions",
    }


def test_build_html_context_defaults_three_part_risk_summary() -> None:
    context = Phase1CodebookReportContextBuilder().build(_payload())

    assert context["risk_summary"] == {
        "has_findings": False,
        "severity": "info",
        "badge_class": "pass",
        "finding_count": "0",
        "primary_risk": "未发现阻断或警戒级风险。",
        "inspection_target": (
            "无需优先 drill-down；保留 hard gate、per-code 和 drift 常规审计记录。"
        ),
        "recommendation": (
            "当前 checkpoint 可按 hard gate 和 selector 结果进入后续候选流程。"
        ),
    }


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


def test_build_html_context_includes_full_code_diagnostic_fields() -> None:
    context = Phase1CodebookReportContextBuilder().build(_payload())

    assert context["code_diagnostics"][0] == {
        "code_id": "3",
        "support": "24",
        "occupancy": "0.125",
        "dominant_morphology": "trend<up>",
        "dominant_morphology_ratio": "0.67",
        "morphology_lift": "-",
        "dominant_motif": "hold",
        "dominant_motif_ratio": "0.5",
        "dominant_pair": "trend-hold",
        "dominant_pair_ratio": "0.45",
        "decoded_mean_advantage": "1.25",
        "decoded_win_rate": "0.55",
        "retention_ratio": "-",
        "fee_drag": "0.02",
        "status": "weak<script>",
        "badge_class": "warn",
    }


def test_build_html_context_includes_oracle_profitability_kpis() -> None:
    context = Phase1CodebookReportContextBuilder().build(_payload())

    assert context["oracle_profitability_kpis"] == [
        {
            "key": "mean_decoded_advantage_vs_flat",
            "label": "mean decoded advantage",
            "value": "0.174",
        },
        {
            "key": "random_label_relative_lift",
            "label": "vs random uplift",
            "value": "0.28",
        },
        {
            "key": "top_5_contribution",
            "label": "top 5% contribution",
            "value": "0.42",
        },
        {
            "key": "trimmed_decoded_advantage",
            "label": "trimmed advantage",
            "value": "0.091",
        },
    ]


def test_build_html_context_includes_per_code_profit_series() -> None:
    context = Phase1CodebookReportContextBuilder().build(_payload())

    assert context["per_code_profit_series"] == [
        {
            "code_id": "3",
            "label": "code 3",
            "value": "1.25",
            "badge_class": "pass",
        }
    ]


def test_build_html_context_includes_oracle_cumulative_return_series() -> None:
    context = Phase1CodebookReportContextBuilder().build(_payload())

    assert context["oracle_cumulative_return_series"] == [
        {
            "key": "dp",
            "label": "DP",
            "points": [
                {"step": "0", "value": "0"},
                {"step": "1", "value": "0.3"},
                {"step": "2", "value": "0.2"},
                {"step": "3", "value": "0.4"},
            ],
        },
        {
            "key": "decoded",
            "label": "Decoded",
            "points": [
                {"step": "0", "value": "0"},
                {"step": "1", "value": "0.2"},
                {"step": "2", "value": "0.15"},
                {"step": "3", "value": "0.25"},
            ],
        },
        {
            "key": "random_label",
            "label": "Random label",
            "points": [
                {"step": "0", "value": "0"},
                {"step": "1", "value": "0.05"},
                {"step": "2", "value": "0.03"},
                {"step": "3", "value": "0.04"},
            ],
        },
        {
            "key": "flat",
            "label": "Flat",
            "points": [
                {"step": "0", "value": "0"},
                {"step": "1", "value": "0"},
                {"step": "2", "value": "0"},
                {"step": "3", "value": "0"},
            ],
        },
    ]
