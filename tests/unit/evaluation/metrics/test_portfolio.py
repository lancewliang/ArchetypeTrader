"""Portfolio metrics 单元测试。

测试用例:
- net_return / sharpe / sortino / MDD / calmar 正确。
- turnover / boundary_cost / cost_paid 正确。
- equity_curve_summary 结构正确。
"""
import pytest

from src.evaluation.metrics.portfolio import (
    build_equity_curve_summary,
    compute_boundary_cost,
    compute_calmar,
    compute_max_drawdown,
    compute_net_return,
    compute_sharpe,
    compute_sortino,
    compute_turnover,
)


class TestPortfolioMetrics:

    def test_net_return(self):
        assert compute_net_return([0.1, -0.05, 0.2]) == pytest.approx(0.25)

    def test_net_return_empty(self):
        assert compute_net_return([]) == 0.0

    def test_sharpe_positive(self):
        returns = [0.01] * 100
        sharpe = compute_sharpe(returns, annualization_factor=252)
        assert sharpe > 0

    def test_sharpe_zero_std(self):
        """常数 return 时 std=0，sharpe=0。"""
        returns = [0.0] * 10
        assert compute_sharpe(returns) == 0.0

    def test_sortino_only_downside(self):
        returns = [0.01, -0.02, 0.01, -0.03, 0.01]
        sortino = compute_sortino(returns, annualization_factor=252)
        # sortino 应该是有限值
        assert isinstance(sortino, float)

    def test_max_drawdown_no_drawdown(self):
        """单调上涨无回撤。"""
        equity = [1.0, 1.1, 1.2, 1.3]
        assert compute_max_drawdown(equity) == pytest.approx(0.0)

    def test_max_drawdown_with_drawdown(self):
        equity = [1.0, 1.2, 0.8, 1.0]
        mdd = compute_max_drawdown(equity)
        # 从 1.2 跌到 0.8，回撤 = (1.2-0.8)/1.2 = 0.333
        assert mdd == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_max_drawdown_empty(self):
        assert compute_max_drawdown([]) == 0.0

    def test_calmar(self):
        assert compute_calmar(0.1, 0.05) == pytest.approx(2.0)

    def test_calmar_zero_dd(self):
        assert compute_calmar(0.1, 0.0) == 0.0

    def test_turnover(self):
        actions = [0, 0, 1, 1, 2, 0]
        # 切换: 0→1, 1→2, 2→0 = 3 次 / 5 间隔
        assert compute_turnover(actions) == pytest.approx(3.0 / 5.0)

    def test_turnover_no_switch(self):
        actions = [1, 1, 1]
        assert compute_turnover(actions) == 0.0

    def test_boundary_cost(self):
        assert compute_boundary_cost([0.01, 0.02, 0.03]) == pytest.approx(0.06)

    def test_equity_curve_summary_structure(self):
        """equity_curve_summary 结构正确。"""
        rewards = [0.1, -0.05, 0.2, -0.1, 0.15]
        summary = build_equity_curve_summary(rewards)
        assert summary.start_value == 1.0
        assert len(summary.per_horizon_cumulative_pnl) == 5
        assert summary.max_value >= summary.min_value
        assert summary.peak_step >= 0
        assert summary.valley_step >= 0

    def test_equity_curve_summary_empty(self):
        summary = build_equity_curve_summary([])
        assert summary.start_value == 1.0
        assert summary.per_horizon_cumulative_pnl == []
