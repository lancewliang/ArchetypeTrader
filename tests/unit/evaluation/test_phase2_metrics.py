"""Phase II metrics 门面单元测试。"""
from __future__ import annotations

from src.evaluation.phase2_metrics import (
    compute_phase2_composite_score,
    phase2_composite_metrics,
    phase2_composite_score_sensitivity,
)


class TestPhase2CompositeMetrics:

    def test_composite_metrics_all_fields(self):
        """phase2_composite_metrics 返回所有必要字段。"""
        records = [
            {"chosen_code": 0, "reward_raw": 1.0, "boundary_cost": 0.1, "cost_paid": 0.2, "step_returns": [0.5, 0.5]},
            {"chosen_code": 1, "reward_raw": -0.5, "boundary_cost": 0.0, "cost_paid": 0.1, "step_returns": [-0.5]},
        ]
        metrics = phase2_composite_metrics(records, {"policy_loss": 0.1}, 3, [False, False, True])
        for key in [
            "net_return",
            "sharpe_ratio",
            "max_drawdown",
            "turnover",
            "action_dominance_ratio",
            "active_archetype_ratio",
            "equity_curve_summary",
            "phase2_composite_score",
        ]:
            assert key in metrics

    def test_composite_score_records_missing_metrics(self):
        score, debug = compute_phase2_composite_score({"net_return": 1.0}, {"net_return": 1.0, "missing": 2.0})
        assert score == 1.0
        assert debug["missing_metrics"] == ["missing"]

    def test_sensitivity_outputs_stability_payload(self):
        result = phase2_composite_score_sensitivity(
            [{"update_idx": 1, "net_return": 1.0}, {"update_idx": 2, "net_return": 2.0}],
            {"net_return": 1.0},
            [-0.2, 0.2],
        )
        assert result["base_best"]["update_idx"] == 2
        assert result["best_update_indices"]
