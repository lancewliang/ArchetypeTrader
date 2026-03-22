"""评估指标单元测试 — ArchetypeTrader

测试 EvaluationEngine 的所有指标计算，包括：
- 已知收益序列的指标计算
- 除零边界情况
- 空收益序列错误处理

需求: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
"""

import math

import numpy as np
import pytest

from src.evaluation.metrics import EvaluationEngine


@pytest.fixture
def engine():
    """默认年化因子 m=52560 的评估引擎。"""
    return EvaluationEngine(annualization_factor=52560)


# ---------------------------------------------------------------------------
# 1. 已知收益序列 — 手工计算验证
# ---------------------------------------------------------------------------

class TestKnownReturns:
    """使用 [0.01, -0.02, 0.03, -0.01, 0.02] 验证各指标。"""

    RETURNS = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    M = 52560

    def test_total_return(self, engine):
        """TR = Π(1 + r_t) - 1"""
        expected = np.prod(1.0 + self.RETURNS) - 1.0
        result = engine.compute_total_return(self.RETURNS)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_annual_volatility(self, engine):
        """AVOL = std(r) × √m"""
        expected = float(np.std(self.RETURNS)) * math.sqrt(self.M)
        result = engine.compute_annual_volatility(self.RETURNS)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_max_drawdown(self, engine):
        """MDD = max((peak - trough) / peak) on cumulative wealth curve."""
        wealth = np.cumprod(1.0 + self.RETURNS)
        running_max = np.maximum.accumulate(wealth)
        drawdowns = (running_max - wealth) / running_max
        expected = float(np.max(drawdowns))
        result = engine.compute_max_drawdown(self.RETURNS)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_annual_sharpe_ratio(self, engine):
        """ASR = mean(r) / std(r) × √m"""
        mean_r = float(np.mean(self.RETURNS))
        std_r = float(np.std(self.RETURNS))
        expected = mean_r / std_r * math.sqrt(self.M)
        result = engine.compute_annual_sharpe_ratio(self.RETURNS)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_annual_calmar_ratio(self, engine):
        """ACR = mean(r) / MDD × m"""
        mean_r = float(np.mean(self.RETURNS))
        wealth = np.cumprod(1.0 + self.RETURNS)
        running_max = np.maximum.accumulate(wealth)
        mdd = float(np.max((running_max - wealth) / running_max))
        expected = mean_r / mdd * self.M
        result = engine.compute_annual_calmar_ratio(self.RETURNS)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_annual_sortino_ratio(self, engine):
        """ASoR = mean(r) / std(negative_returns) × √m"""
        mean_r = float(np.mean(self.RETURNS))
        neg = self.RETURNS[self.RETURNS < 0]
        dd = float(np.std(neg))
        expected = mean_r / dd * math.sqrt(self.M)
        result = engine.compute_annual_sortino_ratio(self.RETURNS)
        assert result == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# 2. 全正收益 — MDD ≈ 0, Sortino = 0.0 (无负收益)
# ---------------------------------------------------------------------------

class TestAllPositiveReturns:

    RETURNS = np.array([0.01, 0.02, 0.03, 0.01, 0.02])

    def test_mdd_zero(self, engine):
        """全正收益时 MDD 应为 0（财富曲线单调递增）。"""
        result = engine.compute_max_drawdown(self.RETURNS)
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_sortino_zero_no_negative(self, engine):
        """无负收益时 Sortino 返回 0.0。"""
        result = engine.compute_annual_sortino_ratio(self.RETURNS)
        assert result == 0.0

    def test_calmar_zero_when_mdd_zero(self, engine):
        """MDD=0 时 Calmar 返回 0.0。"""
        result = engine.compute_annual_calmar_ratio(self.RETURNS)
        assert result == 0.0


# ---------------------------------------------------------------------------
# 3. 全负收益 — TR < 0, MDD 较大
# ---------------------------------------------------------------------------

class TestAllNegativeReturns:

    RETURNS = np.array([-0.01, -0.02, -0.03, -0.01, -0.02])

    def test_total_return_negative(self, engine):
        result = engine.compute_total_return(self.RETURNS)
        assert result < 0.0

    def test_mdd_large(self, engine):
        """全负收益时 MDD 应较大。"""
        result = engine.compute_max_drawdown(self.RETURNS)
        assert result > 0.0
        # 手工验证: 财富曲线持续下降
        wealth = np.cumprod(1.0 + self.RETURNS)
        running_max = np.maximum.accumulate(wealth)
        expected_mdd = float(np.max((running_max - wealth) / running_max))
        assert result == pytest.approx(expected_mdd, rel=1e-10)


# ---------------------------------------------------------------------------
# 4. 常数收益 — σ=0 → Sharpe=0
# ---------------------------------------------------------------------------

class TestConstantReturns:

    RETURNS = np.array([0.01, 0.01, 0.01, 0.01, 0.01])

    def test_sharpe_zero_when_std_zero(self, engine):
        """σ=0 时 Sharpe 返回 0.0。"""
        result = engine.compute_annual_sharpe_ratio(self.RETURNS)
        assert result == 0.0

    def test_volatility_zero(self, engine):
        result = engine.compute_annual_volatility(self.RETURNS)
        assert result == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# 5. 单个收益值 — 边界情况
# ---------------------------------------------------------------------------

class TestSingleReturn:

    def test_single_positive(self, engine):
        r = np.array([0.05])
        assert engine.compute_total_return(r) == pytest.approx(0.05, rel=1e-10)
        # std of single element is 0
        assert engine.compute_annual_sharpe_ratio(r) == 0.0
        # MDD: wealth=[1.05], running_max=[1.05], drawdown=[0]
        assert engine.compute_max_drawdown(r) == pytest.approx(0.0, abs=1e-15)

    def test_single_negative(self, engine):
        r = np.array([-0.05])
        assert engine.compute_total_return(r) == pytest.approx(-0.05, rel=1e-10)
        # MDD: wealth=[0.95], peak=0.95, drawdown=0 (only one point)
        assert engine.compute_max_drawdown(r) == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# 6. 全零收益
# ---------------------------------------------------------------------------

class TestZeroReturns:

    RETURNS = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    def test_total_return_zero(self, engine):
        assert engine.compute_total_return(self.RETURNS) == pytest.approx(0.0, abs=1e-15)

    def test_volatility_zero(self, engine):
        assert engine.compute_annual_volatility(self.RETURNS) == pytest.approx(0.0, abs=1e-15)

    def test_sharpe_zero(self, engine):
        assert engine.compute_annual_sharpe_ratio(self.RETURNS) == 0.0

    def test_mdd_zero(self, engine):
        assert engine.compute_max_drawdown(self.RETURNS) == pytest.approx(0.0, abs=1e-15)

    def test_calmar_zero(self, engine):
        assert engine.compute_annual_calmar_ratio(self.RETURNS) == 0.0

    def test_sortino_zero(self, engine):
        assert engine.compute_annual_sortino_ratio(self.RETURNS) == 0.0


# ---------------------------------------------------------------------------
# 7. 大收益值 — 数值稳定性
# ---------------------------------------------------------------------------

class TestLargeReturns:

    RETURNS = np.array([0.5, -0.3, 0.8, -0.4, 0.6])

    def test_total_return_finite(self, engine):
        result = engine.compute_total_return(self.RETURNS)
        assert np.isfinite(result)

    def test_all_metrics_finite(self, engine):
        metrics = engine.evaluate(self.RETURNS)
        for key, val in metrics.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"


# ---------------------------------------------------------------------------
# 8. evaluate() 返回完整字典
# ---------------------------------------------------------------------------

class TestEvaluateDict:

    EXPECTED_KEYS = {
        "total_return",
        "annual_volatility",
        "max_drawdown",
        "annual_sharpe_ratio",
        "annual_calmar_ratio",
        "annual_sortino_ratio",
    }

    def test_evaluate_returns_all_keys(self, engine):
        r = np.array([0.01, -0.02, 0.03])
        result = engine.evaluate(r)
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_evaluate_values_match_individual(self, engine):
        r = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        result = engine.evaluate(r)
        assert result["total_return"] == pytest.approx(engine.compute_total_return(r))
        assert result["annual_volatility"] == pytest.approx(engine.compute_annual_volatility(r))
        assert result["max_drawdown"] == pytest.approx(engine.compute_max_drawdown(r))
        assert result["annual_sharpe_ratio"] == pytest.approx(engine.compute_annual_sharpe_ratio(r))
        assert result["annual_calmar_ratio"] == pytest.approx(engine.compute_annual_calmar_ratio(r))
        assert result["annual_sortino_ratio"] == pytest.approx(engine.compute_annual_sortino_ratio(r))


# ---------------------------------------------------------------------------
# 9. 自定义年化因子
# ---------------------------------------------------------------------------

class TestCustomAnnualizationFactor:

    def test_custom_factor_volatility(self):
        engine = EvaluationEngine(annualization_factor=252)
        r = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        expected = float(np.std(r)) * math.sqrt(252)
        assert engine.compute_annual_volatility(r) == pytest.approx(expected, rel=1e-10)

    def test_custom_factor_sharpe(self):
        engine = EvaluationEngine(annualization_factor=252)
        r = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        mean_r = float(np.mean(r))
        std_r = float(np.std(r))
        expected = mean_r / std_r * math.sqrt(252)
        assert engine.compute_annual_sharpe_ratio(r) == pytest.approx(expected, rel=1e-10)

    def test_custom_factor_calmar(self):
        engine = EvaluationEngine(annualization_factor=252)
        r = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        mean_r = float(np.mean(r))
        wealth = np.cumprod(1.0 + r)
        running_max = np.maximum.accumulate(wealth)
        mdd = float(np.max((running_max - wealth) / running_max))
        expected = mean_r / mdd * 252
        assert engine.compute_annual_calmar_ratio(r) == pytest.approx(expected, rel=1e-10)

    def test_custom_factor_sortino(self):
        engine = EvaluationEngine(annualization_factor=252)
        r = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        mean_r = float(np.mean(r))
        neg = r[r < 0]
        dd = float(np.std(neg))
        expected = mean_r / dd * math.sqrt(252)
        assert engine.compute_annual_sortino_ratio(r) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Property-Based Tests (PBT) — Property 23: 评估指标公式正确性
# Feature: archetype-trader, Property 23: 评估指标公式正确性
# **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6**
# ---------------------------------------------------------------------------

import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays as st_arrays


# Strategy: random returns array with values in (-0.5, 0.5), length 2..100
_returns_strategy = st_arrays(
    dtype=np.float64,
    shape=st.integers(min_value=2, max_value=100),
    elements=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
)

# Strategy: annualization factor
_ann_factor_strategy = st.integers(min_value=1, max_value=100000)


class TestPBTTotalReturn:
    """PBT: TR = Π(1 + r_t) - 1"""

    @given(returns=_returns_strategy)
    @settings(max_examples=100)
    def test_tr_formula(self, returns):
        engine = EvaluationEngine(annualization_factor=52560)
        expected = float(np.prod(1.0 + returns) - 1.0)
        result = engine.compute_total_return(returns)
        assert result == pytest.approx(expected, rel=1e-9, abs=1e-15)


class TestPBTAnnualVolatility:
    """PBT: AVOL = std(r) × √m"""

    @given(returns=_returns_strategy, m=_ann_factor_strategy)
    @settings(max_examples=100)
    def test_avol_formula(self, returns, m):
        engine = EvaluationEngine(annualization_factor=m)
        expected = float(np.std(returns)) * math.sqrt(m)
        result = engine.compute_annual_volatility(returns)
        assert result == pytest.approx(expected, rel=1e-9, abs=1e-15)


class TestPBTMDDNonNegativity:
    """PBT: MDD ≥ 0 for any returns sequence."""

    @given(returns=_returns_strategy)
    @settings(max_examples=100)
    def test_mdd_non_negative(self, returns):
        engine = EvaluationEngine(annualization_factor=52560)
        result = engine.compute_max_drawdown(returns)
        assert result >= 0.0


class TestPBTMDDBoundedByOne:
    """PBT: MDD ≤ 1 for returns > -1."""

    @given(
        returns=st_arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=100),
            elements=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=100)
    def test_mdd_bounded_by_one(self, returns):
        # All returns > -1 (since min_value=-0.5), so wealth never goes negative
        engine = EvaluationEngine(annualization_factor=52560)
        result = engine.compute_max_drawdown(returns)
        assert result <= 1.0


class TestPBTAnnualSharpeRatio:
    """PBT: ASR = mean(r) / std(r) × √m when std > 0; ASR = 0 when std = 0."""

    @given(returns=_returns_strategy, m=_ann_factor_strategy)
    @settings(max_examples=100)
    def test_asr_formula(self, returns, m):
        engine = EvaluationEngine(annualization_factor=m)
        std_r = float(np.std(returns))
        result = engine.compute_annual_sharpe_ratio(returns)
        if std_r == 0.0:
            assert result == 0.0
        else:
            mean_r = float(np.mean(returns))
            expected = mean_r / std_r * math.sqrt(m)
            assert result == pytest.approx(expected, rel=1e-9, abs=1e-15)


class TestPBTAnnualCalmarRatio:
    """PBT: ACR = mean(r) / MDD × m when MDD > 0; ACR = 0 when MDD = 0."""

    @given(returns=_returns_strategy, m=_ann_factor_strategy)
    @settings(max_examples=100)
    def test_acr_formula(self, returns, m):
        engine = EvaluationEngine(annualization_factor=m)
        mdd = engine.compute_max_drawdown(returns)
        result = engine.compute_annual_calmar_ratio(returns)
        if mdd == 0.0:
            assert result == 0.0
        else:
            mean_r = float(np.mean(returns))
            expected = mean_r / mdd * m
            assert result == pytest.approx(expected, rel=1e-9, abs=1e-15)


class TestPBTAnnualSortinoRatio:
    """PBT: ASoR = mean(r) / DD × √m when DD > 0; ASoR = 0 when no negative returns or DD = 0."""

    @given(returns=_returns_strategy, m=_ann_factor_strategy)
    @settings(max_examples=100)
    def test_asor_formula(self, returns, m):
        engine = EvaluationEngine(annualization_factor=m)
        negative_returns = returns[returns < 0]
        result = engine.compute_annual_sortino_ratio(returns)
        if len(negative_returns) == 0:
            assert result == 0.0
        else:
            dd = float(np.std(negative_returns))
            if dd == 0.0:
                assert result == 0.0
            else:
                mean_r = float(np.mean(returns))
                expected = mean_r / dd * math.sqrt(m)
                assert result == pytest.approx(expected, rel=1e-9, abs=1e-15)


class TestPBTEvaluateCompleteness:
    """PBT: evaluate() returns dict with exactly 6 keys and all values are finite floats."""

    EXPECTED_KEYS = {
        "total_return",
        "annual_volatility",
        "max_drawdown",
        "annual_sharpe_ratio",
        "annual_calmar_ratio",
        "annual_sortino_ratio",
    }

    @given(returns=_returns_strategy)
    @settings(max_examples=100)
    def test_evaluate_completeness(self, returns):
        engine = EvaluationEngine(annualization_factor=52560)
        result = engine.evaluate(returns)
        assert set(result.keys()) == self.EXPECTED_KEYS
        assert len(result) == 6
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not a float: {type(val)}"
            assert np.isfinite(val), f"{key} is not finite: {val}"
