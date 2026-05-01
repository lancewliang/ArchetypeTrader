"""Phase II failure case report 单元测试。"""
from __future__ import annotations

from src.evaluation.diagnostics.phase2_failure_case_report import (
    Phase2FailureCaseReportBuilder,
)


def _records():
    return [
        {"sample_id": "a", "chosen_code": 0, "reward_raw": -2.0, "cost_paid": 0.1},
        {"sample_id": "b", "chosen_code": 1, "reward_raw": 1.0, "cost_paid": 3.0},
        {"sample_id": "c", "chosen_code": 2, "reward_raw": 0.5, "cost_paid": 0.2, "risk_triggered": True},
    ]


class TestPhase2FailureCaseReport:

    def test_worst_return_cases(self):
        builder = Phase2FailureCaseReportBuilder(top_k=1)
        cases = builder.select_cases(_records())
        assert any(c.category == "worst_return" and c.sample_id == "a" for c in cases)

    def test_largest_cost_cases(self):
        builder = Phase2FailureCaseReportBuilder(top_k=1)
        cases = builder.select_cases(_records())
        assert any(c.category == "largest_cost" and c.sample_id == "b" for c in cases)

    def test_unstable_switching_cases_and_writes_files(self, tmp_path):
        builder = Phase2FailureCaseReportBuilder(top_k=2)
        cases = builder.select_cases(_records())
        assert any(c.category == "unstable_switching" for c in cases)
        assert builder.write_jsonl(cases, tmp_path / "cases.jsonl").exists()
        assert builder.write_html(cases, tmp_path / "cases.html").exists()
