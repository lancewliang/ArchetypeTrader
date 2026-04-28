"""评估模块单元测试

测试重构后的 PortfolioTracker、generate_base_actions、compute_base_return、
run_horizon_inference 等核心逻辑，使用 TradingEnv 进行集成验证。
"""

import numpy as np
import polars as pl
import pytest
import torch

from src.config import Config
from src.data.feature_pipeline import FIXED_FEATURES
from src.env.trading_env import TradingEnv
from src.evaluation.portfolio_tracker import (
    PortfolioTracker,
    PortfolioState,
    StepRecord,
    ACTION_LABELS,
)
from src.evaluation.inference_runner import (
    generate_base_actions,
    generate_current_base_action,
    compute_base_return,
    run_horizon_inference,
    run_online_horizon_inference,
)
from src.phase1.codebook import VQCodebook
from src.phase1.vq_decoder import VQDecoder
from src.phase2.selection_agent import SelectionAgent
from src.phase3.policy_adapter import PolicyAdapter
from src.phase3.refinement_agent import RefinementAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STATE_DIM = len(FIXED_FEATURES)
HORIZON = 12
NUM_ARCHETYPES = 4
LATENT_DIM = 8


def _make_env(
    T: int = 48,
    pair: str = "ETH",
    horizon: int = HORIZON,
    price_start: float = 2000.0,
    price_step: float = 1.0,
) -> TradingEnv:
    """创建测试用 TradingEnv。"""
    rng = np.random.RandomState(42)
    states = rng.randn(T, STATE_DIM).astype(np.float64)
    prices = np.arange(price_start, price_start + T * price_step, price_step)[:T]
    feature_cols = FIXED_FEATURES
    states_df = pl.DataFrame(states, schema=feature_cols[:STATE_DIM])
    return TradingEnv(
        states=states, prices=prices, pair=pair, horizon=horizon,
        states_dataframe=states_df,
    )


@pytest.fixture
def env():
    return _make_env()


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def decoder(device):
    d = VQDecoder(state_dim=STATE_DIM, code_dim=LATENT_DIM, hidden_dim=32, action_dim=3).to(device)
    d.eval()
    return d


@pytest.fixture
def codebook(device):
    cb = VQCodebook(num_codes=NUM_ARCHETYPES, code_dim=LATENT_DIM).to(device)
    cb.eval()
    return cb


@pytest.fixture
def selection_agent(device):
    sa = SelectionAgent(state_dim=STATE_DIM, num_archetypes=NUM_ARCHETYPES).to(device)
    sa.eval()
    return sa


@pytest.fixture
def refinement_agent(device):
    context_dim = LATENT_DIM + 3
    ra = RefinementAgent(market_dim=STATE_DIM, context_dim=context_dim).to(device)
    ra.eval()
    return ra


# ===========================================================================
# PortfolioTracker 单元测试
# ===========================================================================

class TestPortfolioState:
    """PortfolioState dataclass 基本行为。"""

    def test_defaults(self):
        s = PortfolioState()
        assert s.cash == 0.0
        assert s.position == 0
        assert s.short_open_value == 0.0
        assert s.avg_hold_price == 0.0


class TestStepRecord:
    """StepRecord 序列化。"""

    def test_to_dict_rounds_floats(self):
        r = StepRecord(
            state_index=5, action=2, action_label="long",
            execution_price=1234.5678901, cash=9999.9999999,
        )
        d = r.to_dict()
        assert d["execution_price"] == round(1234.5678901, 6)
        assert d["cash"] == round(9999.9999999, 6)
        assert d["state_index"] == 5


class TestPortfolioTrackerInit:
    """PortfolioTracker 初始化。"""

    def test_initial_capital(self):
        t = PortfolioTracker(initial_capital=10000.0)
        assert t.state.cash == 10000.0
        assert t.initial_capital == 10000.0
        assert t.records == []


class TestSettlePreviousHorizon:
    """跨 horizon 平仓清算逻辑。"""

    def test_no_position_no_settle(self):
        t = PortfolioTracker(1000.0)
        t.settle_previous_horizon(price=100.0, t_start=0)
        assert len(t.records) == 0
        assert t.state.cash == 1000.0

    def test_long_position_settle_no_action(self):
        """未提供 new_first_action 时退化为强制平仓（向后兼容）。"""
        t = PortfolioTracker(10000.0)
        t.state.position = 5
        t.state.avg_hold_price = 90.0
        t.settle_previous_horizon(price=100.0, t_start=12)
        assert len(t.records) == 1
        assert t.records[0]["action"] == "settle"
        assert t.state.position == 0
        assert t.state.avg_hold_price == 0.0
        # 卖出 5 × 100 = 500 回收到 cash（无手续费，因为 commission_rate 默认 0）
        assert t.state.cash == pytest.approx(10000.0 + 500.0)

    def test_short_position_settle_no_action(self):
        """未提供 new_first_action 时退化为强制平仓（向后兼容）。"""
        t = PortfolioTracker(10000.0)
        t.state.position = -5
        t.state.short_open_value = 500.0  # 开仓时 5 × 100
        t.settle_previous_horizon(price=90.0, t_start=12)
        assert len(t.records) == 1
        assert t.records[0]["action"] == "settle"
        assert t.state.position == 0
        assert t.state.short_open_value == 0.0
        # 空头平仓盈亏 = sov - close_debt = 500 - 5*90 = 50
        assert t.state.cash == pytest.approx(10050.0)

    def test_long_to_long_skip_settle(self):
        """多→多: 同方向延续，跳过平仓。"""
        t = PortfolioTracker(10000.0)
        t.state.position = 5
        t.state.avg_hold_price = 90.0
        t.state.cash = 9550.0  # 买入花了 450
        t.settle_previous_horizon(
            price=100.0, t_start=12,
            new_first_action=2, m=5, commission_rate=0.0002,
        )
        # 不应产生任何记录
        assert len(t.records) == 0
        # 持仓和 cash 不变
        assert t.state.position == 5
        assert t.state.avg_hold_price == 90.0
        assert t.state.cash == 9550.0

    def test_short_to_short_skip_settle(self):
        """空→空: 同方向延续，跳过平仓。"""
        t = PortfolioTracker(10000.0)
        t.state.position = -5
        t.state.short_open_value = 500.0
        t.state.cash = 10000.0
        t.settle_previous_horizon(
            price=90.0, t_start=12,
            new_first_action=0, m=5, commission_rate=0.0002,
        )
        assert len(t.records) == 0
        assert t.state.position == -5
        assert t.state.short_open_value == 500.0
        assert t.state.cash == 10000.0

    def test_long_to_short_settle_with_fee(self):
        """多→空: 方向改变，平仓并收取手续费+滑点。"""
        t = PortfolioTracker(10000.0)
        t.state.position = 5
        t.state.avg_hold_price = 90.0
        commission_rate = 0.0002
        price = 100.0
        slippage = 0.50
        expected_commission = round(commission_rate * 5 * price, 2)  # 0.10
        t.settle_previous_horizon(
            price=price, t_start=12,
            new_first_action=0, m=5, commission_rate=commission_rate,
            slippage=slippage,
        )
        assert len(t.records) == 1
        assert t.records[0]["action"] == "settle"
        assert t.records[0]["commission"] == pytest.approx(expected_commission)
        assert t.records[0]["slippage"] == pytest.approx(slippage)
        assert t.state.position == 0
        # cash = 10000 + 5*100 - 0.10 - 0.50 = 10499.40
        assert t.state.cash == pytest.approx(10000.0 + 500.0 - expected_commission - slippage)

    def test_short_to_long_settle_with_fee(self):
        """空→多: 方向改变，平仓并收取手续费+滑点。"""
        t = PortfolioTracker(10000.0)
        t.state.position = -5
        t.state.short_open_value = 500.0  # 开仓 5 × 100
        commission_rate = 0.0002
        price = 90.0
        slippage = 0.30
        expected_commission = round(commission_rate * 5 * price, 2)  # 0.09
        t.settle_previous_horizon(
            price=price, t_start=12,
            new_first_action=2, m=5, commission_rate=commission_rate,
            slippage=slippage,
        )
        assert len(t.records) == 1
        assert t.records[0]["commission"] == pytest.approx(expected_commission)
        assert t.records[0]["slippage"] == pytest.approx(slippage)
        assert t.state.position == 0
        assert t.state.short_open_value == 0.0
        # pnl = 500 - 5*90 = 50, cash = 10000 + 50 - 0.09 - 0.30
        assert t.state.cash == pytest.approx(10050.0 - expected_commission - slippage)

    def test_long_to_flat_settle_with_fee(self):
        """多→空仓: 平仓并收取手续费。"""
        t = PortfolioTracker(10000.0)
        t.state.position = 5
        t.state.avg_hold_price = 100.0
        commission_rate = 0.0002
        price = 100.0
        expected_commission = round(commission_rate * 5 * price, 2)
        t.settle_previous_horizon(
            price=price, t_start=12,
            new_first_action=1, m=5, commission_rate=commission_rate,
        )
        assert len(t.records) == 1
        assert t.state.position == 0
        assert t.state.cash == pytest.approx(10000.0 + 500.0 - expected_commission)

    def test_short_to_flat_settle_with_fee(self):
        """空→空仓: 平仓并收取手续费。"""
        t = PortfolioTracker(10000.0)
        t.state.position = -5
        t.state.short_open_value = 500.0
        commission_rate = 0.0002
        price = 100.0
        expected_commission = round(commission_rate * 5 * price, 2)
        t.settle_previous_horizon(
            price=price, t_start=12,
            new_first_action=1, m=5, commission_rate=commission_rate,
        )
        assert len(t.records) == 1
        assert t.state.position == 0
        # pnl = 500 - 500 = 0, cash = 10000 + 0 - 0.10
        assert t.state.cash == pytest.approx(10000.0 - expected_commission)

    def test_short_settle_then_open_short_cash_correct(self):
        """复现 bug: settle 空头后紧接着开空头，cash 不应被额外扣减。
        使用 new_first_action=0 (short) 时应跳过平仓。"""
        initial_capital = 187481.0
        t = PortfolioTracker(initial_capital)
        t.state.position = -100
        t.state.short_open_value = 188601.0  # 100 × 1886.01
        t.state.cash = initial_capital

        # 新 horizon 首个动作也是 short → 跳过平仓
        t.settle_previous_horizon(
            price=1878.6, t_start=72,
            new_first_action=0, m=100, commission_rate=0.0002,
        )
        # 同方向延续，不平仓
        assert t.state.position == -100
        assert t.state.cash == pytest.approx(initial_capital)
        assert t.state.short_open_value == pytest.approx(188601.0)


class TestUpdateCashForTrade:
    """单步资金更新逻辑。"""

    def test_no_change(self):
        t = PortfolioTracker(10000.0)
        t.update_cash_for_trade(old_position=0, new_position=0, exec_price=100.0, t=0)
        assert t.state.cash == 10000.0

    def test_flat_to_long(self):
        t = PortfolioTracker(10000.0)
        t.update_cash_for_trade(old_position=0, new_position=5, exec_price=100.0, t=0)
        # 买入 5 × 100 = 500
        assert t.state.cash == pytest.approx(10000.0 - 500.0)

    def test_long_to_flat(self):
        t = PortfolioTracker(10000.0)
        t.state.cash = 9500.0  # 之前买入花了 500
        t.update_cash_for_trade(old_position=5, new_position=0, exec_price=110.0, t=1)
        # 卖出 5 × 110 = 550 回收
        assert t.state.cash == pytest.approx(9500.0 + 550.0)

    def test_flat_to_short(self):
        t = PortfolioTracker(10000.0)
        t.update_cash_for_trade(old_position=0, new_position=-5, exec_price=100.0, t=0)
        # 做空不影响 cash，只记录 short_open_value
        assert t.state.cash == 10000.0
        assert t.state.short_open_value == pytest.approx(500.0)

    def test_short_to_flat(self):
        t = PortfolioTracker(10000.0)
        t.state.short_open_value = 500.0  # 开仓 5 × 100
        t.update_cash_for_trade(old_position=-5, new_position=0, exec_price=90.0, t=1)
        # 平仓盈亏 = 500 - 5*90 = 50 (价格跌了赚了)
        assert t.state.cash == pytest.approx(10050.0)
        assert t.state.short_open_value == 0.0


class TestPositionChangePnl:
    """盈亏计算纯函数。"""

    def test_long_close_profit(self):
        pnl = PortfolioTracker.compute_position_change_pnl(
            old_position=5, new_position=0, exec_price=110.0, avg_hold_price=100.0,
        )
        assert pnl == pytest.approx(50.0)

    def test_long_close_loss(self):
        pnl = PortfolioTracker.compute_position_change_pnl(
            old_position=5, new_position=0, exec_price=90.0, avg_hold_price=100.0,
        )
        assert pnl == pytest.approx(-50.0)

    def test_short_close_profit(self):
        pnl = PortfolioTracker.compute_position_change_pnl(
            old_position=-5, new_position=0, exec_price=90.0, avg_hold_price=100.0,
        )
        assert pnl == pytest.approx(50.0)

    def test_no_change_zero_pnl(self):
        pnl = PortfolioTracker.compute_position_change_pnl(
            old_position=5, new_position=5, exec_price=110.0, avg_hold_price=100.0,
        )
        assert pnl == 0.0

    def test_long_partial_close(self):
        pnl = PortfolioTracker.compute_position_change_pnl(
            old_position=10, new_position=5, exec_price=110.0, avg_hold_price=100.0,
        )
        # 减仓 5 单位，每单位赚 10
        assert pnl == pytest.approx(50.0)


class TestUpdateAvgHoldPrice:
    """平均持仓价格更新。"""

    def test_close_all_resets(self):
        p = PortfolioTracker.update_avg_hold_price(5, 0, 110.0, 100.0)
        assert p == 0.0

    def test_open_from_flat(self):
        p = PortfolioTracker.update_avg_hold_price(0, 5, 100.0, 0.0)
        assert p == 100.0

    def test_add_to_long(self):
        # 原持 5 @ 100，加仓 5 @ 120 → 均价 110
        p = PortfolioTracker.update_avg_hold_price(5, 10, 120.0, 100.0)
        assert p == pytest.approx(110.0)

    def test_reduce_long_keeps_avg(self):
        p = PortfolioTracker.update_avg_hold_price(10, 5, 120.0, 100.0)
        assert p == 100.0  # 减仓不改变均价

    def test_flip_direction(self):
        p = PortfolioTracker.update_avg_hold_price(5, -3, 110.0, 100.0)
        assert p == 110.0  # 翻转方向用当前价


class TestComputeTotalValue:
    """总资产计算。"""

    def test_flat_position(self):
        t = PortfolioTracker(10000.0)
        hv, sd, tv, profit = t.compute_total_value(0, 100.0)
        assert hv == 0.0
        assert sd == 0.0
        assert tv == 10000.0
        assert profit == 0.0

    def test_long_position(self):
        t = PortfolioTracker(10000.0)
        t.state.cash = 9000.0  # 花了 1000 买入
        hv, sd, tv, profit = t.compute_total_value(10, 100.0)
        assert hv == 1000.0
        assert sd == 0.0
        assert tv == 10000.0  # 9000 + 1000
        assert profit == 0.0

    def test_short_position_with_unrealized_pnl(self):
        t = PortfolioTracker(10000.0)
        t.state.short_open_value = 1000.0  # 开仓 10 × 100
        hv, sd, tv, profit = t.compute_total_value(-10, 90.0)
        # short_debt = 10 × 90 = 900
        # unrealized_pnl = 1000 - 900 = 100
        # total = 10000 + 0 + 100 = 10100
        assert hv == 0.0
        assert sd == 900.0
        assert tv == pytest.approx(10100.0)
        assert profit == pytest.approx(100.0)


class TestRecordStep:
    """record_step 集成测试。"""

    def test_generates_record(self):
        t = PortfolioTracker(10000.0)
        t.record_step(
            t=0, a_final=2, exec_price=100.0,
            old_position=0, new_position=5,
            commission=1.0, slippage=0.5,
        )
        assert len(t.records) == 1
        rec = t.records[0]
        assert rec["action"] == 2
        assert rec["action_label"] == "long"
        assert rec["position_after"] == 5
        assert t.state.position == 5
        assert t.state.avg_hold_price == 100.0

    def test_commission_and_slippage_deducted_from_cash(self):
        """手续费和滑点应从 cash 中扣除。"""
        t = PortfolioTracker(10000.0)
        t.update_cash_for_trade(0, 5, 100.0, 0)  # 买入 5×100=500, cash=9500
        t.record_step(
            t=0, a_final=2, exec_price=100.0,
            old_position=0, new_position=5,
            commission=1.0, slippage=0.5,
        )
        # cash = 9500 - 1.0 - 0.5 = 9498.5
        assert t.state.cash == pytest.approx(9498.5)
        rec = t.records[0]
        assert rec["cash"] == pytest.approx(9498.5)
        # total_value = cash + holding = 9498.5 + 500 = 9998.5
        assert rec["total_value"] == pytest.approx(9998.5)
        # profit = 9998.5 - 10000 = -1.5
        assert rec["profit"] == pytest.approx(-1.5)

    def test_zero_fee_no_deduction(self):
        """无费用时 cash 不变。"""
        t = PortfolioTracker(10000.0)
        t.record_step(
            t=0, a_final=1, exec_price=100.0,
            old_position=0, new_position=0,
            commission=0.0, slippage=0.0,
        )
        assert t.state.cash == pytest.approx(10000.0)


# ===========================================================================
# generate_base_actions / compute_base_return 与 TradingEnv 集成测试
# ===========================================================================

class TestGenerateBaseActions:
    """generate_base_actions 与 VQDecoder 集成。"""

    def test_output_shape(self, env, decoder, codebook, device):
        z_q = codebook.embeddings.weight[0].unsqueeze(0)
        horizon_states = env.states[:HORIZON]
        actions = generate_base_actions(decoder, z_q, horizon_states, device)
        assert actions.shape == (HORIZON,)

    def test_actions_in_valid_range(self, env, decoder, codebook, device):
        z_q = codebook.embeddings.weight[0].unsqueeze(0)
        horizon_states = env.states[:HORIZON]
        actions = generate_base_actions(decoder, z_q, horizon_states, device)
        assert np.all((actions >= 0) & (actions <= 2))

    def test_deterministic(self, env, decoder, codebook, device):
        z_q = codebook.embeddings.weight[0].unsqueeze(0)
        horizon_states = env.states[:HORIZON]
        a1 = generate_base_actions(decoder, z_q, horizon_states, device)
        a2 = generate_base_actions(decoder, z_q, horizon_states, device)
        np.testing.assert_array_equal(a1, a2)

    def test_current_action_matches_full_sequence_last_step(self, env, decoder, codebook, device):
        z_q = codebook.embeddings.weight[0].unsqueeze(0)
        horizon_states = env.states[:HORIZON]
        full_actions = generate_base_actions(decoder, z_q, horizon_states, device)

        prefix_states = horizon_states[:7]
        current_action = generate_current_base_action(decoder, z_q, prefix_states, device)

        assert current_action == int(full_actions[6])


class TestComputeBaseReturn:
    """compute_base_return 与 TradingEnv 集成。"""

    def test_returns_float(self, env):
        actions = np.ones(HORIZON, dtype=np.int64)  # 全 flat
        ret = compute_base_return(env, 0, actions)
        assert isinstance(ret, float)

    def test_flat_actions_zero_return(self):
        """全 flat 动作 → 持仓为 0 → reward 应为 0。"""
        env = _make_env(T=24, horizon=12)
        actions = np.ones(12, dtype=np.int64)
        ret = compute_base_return(env, 0, actions)
        assert ret == pytest.approx(0.0)

    def test_long_actions_positive_return_with_rising_prices(self):
        """价格上涨 + 全 long → 正收益。"""
        T = 24
        states = np.random.randn(T, STATE_DIM)
        prices = np.linspace(100, 200, T)
        states_df = pl.DataFrame(states, schema=FIXED_FEATURES[:STATE_DIM])
        env = TradingEnv(states=states, prices=prices, pair="ETH", horizon=12, states_dataframe=states_df)
        actions = np.full(12, 2, dtype=np.int64)  # 全 long
        ret = compute_base_return(env, 0, actions)
        assert ret > 0.0


# ===========================================================================
# run_horizon_inference 与 TradingEnv 集成测试
# ===========================================================================

class TestRunHorizonInference:
    """run_horizon_inference 端到端集成。"""

    def _run_one_horizon(self, env, refinement_agent, device):
        """辅助: 在第 0 个 horizon 上运行一次推理。"""
        codebook = VQCodebook(num_codes=NUM_ARCHETYPES, code_dim=LATENT_DIM).to(device)
        decoder = VQDecoder(state_dim=STATE_DIM, code_dim=LATENT_DIM, hidden_dim=32, action_dim=3).to(device)
        decoder.eval()
        codebook.eval()

        z_q = codebook.embeddings.weight[0].unsqueeze(0)
        e_a_sel = codebook.embeddings.weight[0].detach().cpu().numpy()
        horizon_states = env.states[:HORIZON]
        base_actions = generate_base_actions(decoder, z_q, horizon_states, device)

        initial_capital = float(env.m) * float(env.prices[0])
        tracker = PortfolioTracker(initial_capital)

        step_returns = run_horizon_inference(
            env=env,
            horizon_idx=0,
            base_actions=base_actions,
            refinement_agent=refinement_agent,
            policy_adapter=PolicyAdapter(),
            e_a_sel=e_a_sel,
            device=device,
            horizon=HORIZON,
            tracker=tracker,
        )
        return step_returns, tracker

    def test_returns_list_of_correct_length(self, env, refinement_agent, device):
        step_returns, _ = self._run_one_horizon(env, refinement_agent, device)
        assert isinstance(step_returns, list)
        assert len(step_returns) == HORIZON

    def test_tracker_has_records(self, env, refinement_agent, device):
        _, tracker = self._run_one_horizon(env, refinement_agent, device)
        # 每步都应有一条记录
        assert len(tracker.records) == HORIZON

    def test_records_have_required_fields(self, env, refinement_agent, device):
        _, tracker = self._run_one_horizon(env, refinement_agent, device)
        required = {
            "state_index", "action", "action_label", "execution_price",
            "trade_quantity", "position_after", "avg_hold_price",
            "commission", "slippage", "position_change_pnl",
            "cash", "holding_value", "short_debt", "total_value", "profit", "side",
        }
        for rec in tracker.records:
            assert required.issubset(rec.keys())

    def test_action_labels_valid(self, env, refinement_agent, device):
        _, tracker = self._run_one_horizon(env, refinement_agent, device)
        valid_labels = set(ACTION_LABELS.values())
        for rec in tracker.records:
            assert rec["action_label"] in valid_labels

    def test_state_indices_sequential(self, env, refinement_agent, device):
        _, tracker = self._run_one_horizon(env, refinement_agent, device)
        indices = [r["state_index"] for r in tracker.records]
        assert indices == list(range(0, HORIZON))

    def test_multi_horizon_continuity(self, refinement_agent, device):
        """跨多个 horizon 运行，验证 tracker 状态连续传递。"""
        env = _make_env(T=48, horizon=12)
        codebook = VQCodebook(num_codes=NUM_ARCHETYPES, code_dim=LATENT_DIM).to(device)
        decoder = VQDecoder(state_dim=STATE_DIM, code_dim=LATENT_DIM, hidden_dim=32, action_dim=3).to(device)
        decoder.eval()
        codebook.eval()

        initial_capital = float(env.m) * float(env.prices[0])
        tracker = PortfolioTracker(initial_capital)
        all_returns = []

        for h_idx in range(env.num_horizons):
            start = h_idx * env.horizon
            end = min(start + env.horizon, len(env.states))
            horizon_states = env.states[start:end]

            z_q = codebook.embeddings.weight[0].unsqueeze(0)
            e_a_sel = codebook.embeddings.weight[0].detach().cpu().numpy()
            base_actions = generate_base_actions(decoder, z_q, horizon_states, device)

            step_returns = run_horizon_inference(
                env=env, horizon_idx=h_idx, base_actions=base_actions,
                refinement_agent=refinement_agent, policy_adapter=PolicyAdapter(),
                e_a_sel=e_a_sel, device=device,
                horizon=env.horizon, tracker=tracker,
            )
            all_returns.extend(step_returns)

        # 4 horizons × 12 steps = 48 步
        total_steps = env.num_horizons * env.horizon
        assert len(all_returns) == total_steps
        # tracker 记录可能包含 settle 记录，至少有 total_steps 条
        assert len(tracker.records) >= total_steps

    def test_deterministic_inference(self, env, refinement_agent, device):
        """相同输入应产生相同输出。"""
        torch.manual_seed(0)
        r1, t1 = self._run_one_horizon(env, refinement_agent, device)
        torch.manual_seed(0)
        r2, t2 = self._run_one_horizon(env, refinement_agent, device)
        np.testing.assert_array_almost_equal(r1, r2)

    def test_online_path_matches_predecoded_path_without_refinement(self, env, device):
        codebook = VQCodebook(num_codes=NUM_ARCHETYPES, code_dim=LATENT_DIM).to(device)
        decoder = VQDecoder(state_dim=STATE_DIM, code_dim=LATENT_DIM, hidden_dim=32, action_dim=3).to(device)
        decoder.eval()
        codebook.eval()

        z_q = codebook.embeddings.weight[0].unsqueeze(0)
        e_a_sel = codebook.embeddings.weight[0].detach().cpu().numpy()
        horizon_states = env.states[:HORIZON]
        base_actions = generate_base_actions(decoder, z_q, horizon_states, device)

        initial_capital = float(env.m) * float(env.prices[0])

        tracker_predecoded = PortfolioTracker(initial_capital)
        step_returns_predecoded = run_horizon_inference(
            env=env,
            horizon_idx=0,
            base_actions=base_actions,
            refinement_agent=None,
            policy_adapter=PolicyAdapter(),
            e_a_sel=e_a_sel,
            device=device,
            horizon=HORIZON,
            tracker=tracker_predecoded,
        )

        env_online = _make_env(T=48, horizon=12)
        tracker_online = PortfolioTracker(initial_capital)
        step_returns_online = run_online_horizon_inference(
            env=env_online,
            horizon_idx=0,
            decoder=decoder,
            z_q=z_q,
            refinement_agent=None,
            policy_adapter=PolicyAdapter(),
            e_a_sel=e_a_sel,
            device=device,
            horizon=HORIZON,
            tracker=tracker_online,
        )

        np.testing.assert_allclose(step_returns_online, step_returns_predecoded)
        assert tracker_online.records == tracker_predecoded.records
