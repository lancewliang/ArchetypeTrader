"""``FailureCaseReportBuilder`` 单元测试."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.phase1.evaluation.diagnostics.failure_case_report import FailureCaseReportBuilder


def _record(sample_id: str, **kwargs):
    base = {
        "sample_id": sample_id,
        "student_net_return": 0.0,
        "regret_to_dp": 0.0,
        "cost_paid": 0.0,
        "switch_timing_error": 0.0,
        "teacher_actions": [1, 1, 1, 1],
        "student_actions": [1, 1, 1, 1],
        "teacher_is_no_trade": True,
        "student_turnover": 0,
    }
    base.update(kwargs)
    return base


def test_select_top_k_worst_student_return():
    builder = FailureCaseReportBuilder(top_k=2, max_cases_per_report=10, action_labels={0: "S", 1: "F", 2: "L"})
    records = [
        _record("a", student_net_return=-0.3),
        _record("b", student_net_return=-0.5),
        _record("c", student_net_return=0.2),
    ]
    cases = builder.select_cases(records)
    worst = cases["worst_student_return"]
    assert [c.sample_id for c in worst] == ["b", "a"]


def test_classify_late_entry():
    builder = FailureCaseReportBuilder(top_k=1, max_cases_per_report=1, action_labels={0: "S", 1: "F", 2: "L"})
    records = [_record("a", student_net_return=-0.1, switch_timing_error=10)]
    cases = builder.select_cases(records)
    case = cases["worst_student_return"][0]
    assert "late_entry" in case.failure_modes


def test_html_no_external_dependencies(tmp_path):
    builder = FailureCaseReportBuilder(top_k=1, max_cases_per_report=1, action_labels={0: "S", 1: "F", 2: "L"})
    cases = builder.select_cases([_record("a")])
    out = builder.write_html(cases, tmp_path / "report.html")
    text = out.read_text(encoding="utf-8")
    assert "http" not in text  # 不应包含 cdn 引用


def test_jsonl_round_trip(tmp_path):
    builder = FailureCaseReportBuilder(top_k=1, max_cases_per_report=1, action_labels={0: "S", 1: "F", 2: "L"})
    cases = builder.select_cases([_record("a", student_net_return=-0.1)])
    out = builder.write_jsonl(cases, tmp_path / "cases.jsonl")
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    obj = json.loads(lines[0])
    assert obj["sample_id"] == "a"
