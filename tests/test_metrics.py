import math

import numpy as np
import pandas as pd
import pytest

from screener.analysis.metrics import (
    annualized_volatility,
    beta,
    cagr,
    max_drawdown,
    performance_stats,
    period_return,
)


def series(values):
    return pd.Series(values, index=pd.bdate_range("2021-01-01", periods=len(values)), dtype=float)


def test_max_drawdown_known_path():
    assert max_drawdown(series([100, 120, 60, 90]), window=None) == pytest.approx(-0.5)


def test_period_return_and_short_series():
    s = series([100, 110, 121])
    assert period_return(s, 2) == pytest.approx(0.21)
    assert math.isnan(period_return(s, 5))


def test_volatility_zero_for_constant_growth():
    s = series(100 * 1.001 ** np.arange(300))
    assert annualized_volatility(s) == pytest.approx(0.0, abs=1e-10)


def test_beta_of_scaled_returns():
    rng = np.random.default_rng(0)
    market_ret = rng.normal(0, 0.01, 400)
    market = series(100 * np.cumprod(1 + market_ret))
    double = series(100 * np.cumprod(1 + 2 * market_ret))
    assert beta(market, market) == pytest.approx(1.0)
    assert beta(double, market) == pytest.approx(2.0, rel=1e-6)
    assert math.isnan(beta(market, None))


def test_cagr_doubling_in_one_year():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-12-31"]))
    assert cagr(s) == pytest.approx(1.0, rel=0.01)


def test_performance_stats_keys():
    stats = performance_stats(series(np.full(100, 0.001)))
    assert set(stats) == {"total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown"}
    assert stats["max_drawdown"] == 0
