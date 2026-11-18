import numpy as np
import pytest

from screener.analysis.backtest import desired_positions, run_backtest
from screener.analysis.indicators import add_indicators
from screener.analysis.signals import BUY_ACTIONS, build_trade_plan, decide_action, technical_score
from tests.conftest import make_ohlcv


@pytest.mark.parametrize(
    "tech, opp, rsi, expected",
    [
        (80, 90, 55, "STRONG BUY"),
        (80, 90, 78, "BUY ON PULLBACK"),  # strong but overbought
        (40, 55, 50, "BUY"),
        (40, 30, 50, "HOLD"),  # good timing, unattractive stock
        (0, 70, 50, "WATCH"),  # good stock, no trend yet
        (-40, 90, 50, "SELL"),
        (-80, 20, 50, "STRONG SELL"),
        (float("nan"), 90, 50, "N/A"),
    ],
)
def test_decide_action(config, tech, opp, rsi, expected):
    assert decide_action(tech, opp, rsi, config["signals"]) == expected


def test_uptrend_gives_buy_plan_with_ordered_levels(config, uptrend):
    plan = build_trade_plan(add_indicators(uptrend), 80.0, "Medium", None, config["signals"])
    assert plan.technical_score >= 20
    assert plan.action in BUY_ACTIONS
    assert plan.stop_loss < plan.entry < plan.take_profit
    reward = plan.take_profit - plan.entry
    risk = plan.entry - plan.stop_loss
    assert reward / risk == pytest.approx(config["signals"]["reward_risk"], rel=1e-2)


def test_downtrend_gives_sell(config, downtrend):
    plan = build_trade_plan(add_indicators(downtrend), 80.0, "Medium", None, config["signals"])
    assert plan.action in ("SELL", "STRONG SELL")
    assert plan.entry is None


def test_riskier_stocks_get_wider_stops(config, uptrend):
    ind = add_indicators(uptrend)
    low = build_trade_plan(ind, 80.0, "Low", None, config["signals"])
    high = build_trade_plan(ind, 80.0, "High", None, config["signals"])
    assert high.stop_loss < low.stop_loss


def test_technical_score_range(config, uptrend):
    score = technical_score(add_indicators(uptrend), config["signals"]).dropna()
    assert score.between(-100, 100).all()


def test_backtest_decisions_have_no_lookahead(config, uptrend):
    ind = add_indicators(uptrend)
    score = technical_score(ind, config["signals"]).to_numpy()
    args = (20, -20, 3.0)
    full = desired_positions(ind["close"].to_numpy(), score, ind["atr14"].to_numpy(), *args)
    cut = 450
    partial = desired_positions(
        ind["close"].to_numpy()[:cut], score[:cut], ind["atr14"].to_numpy()[:cut], *args
    )
    np.testing.assert_array_equal(full[:cut], partial)


def test_backtest_on_flat_prices_does_nothing(config):
    flat = make_ohlcv(np.full(500, 50.0))
    result = run_backtest(flat, config["signals"], config["backtest"])
    assert result.trades.empty
    assert result.equity.iloc[-1] == pytest.approx(1.0)


def test_backtest_costs_reduce_returns(config, demo_provider):
    history = demo_provider.get_history(["CHIP.D"], "5y")["CHIP.D"]
    free = run_backtest(history, config["signals"], dict(config["backtest"], cost_bps=0))
    costly = run_backtest(history, config["signals"], dict(config["backtest"], cost_bps=50))
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert len(free.trades) > 0
    assert set(free.stats.columns) == {"Strategy", "Buy & Hold"}
