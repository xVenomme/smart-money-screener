"""Return and risk metrics computed from daily prices."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def period_return(close: pd.Series, days: int) -> float:
    close = close.dropna()
    if len(close) <= days:
        return math.nan
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1)


def annualized_volatility(close: pd.Series, window: int = TRADING_DAYS) -> float:
    returns = close.pct_change().dropna().iloc[-window:]
    if len(returns) < 20:
        return math.nan
    return float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))


def drawdown_series(close_or_equity: pd.Series) -> pd.Series:
    running_max = close_or_equity.cummax()
    return close_or_equity / running_max - 1


def max_drawdown(close: pd.Series, window: int | None = TRADING_DAYS) -> float:
    """Worst peak-to-trough loss (negative number, e.g. -0.25 = -25%)."""
    series = close.dropna()
    if window:
        series = series.iloc[-window:]
    if series.empty:
        return math.nan
    return float(drawdown_series(series).min())


def beta(close: pd.Series, benchmark_close: pd.Series | None, window: int = TRADING_DAYS) -> float:
    """Sensitivity to the benchmark: cov(r, r_m) / var(r_m) on daily returns."""
    if benchmark_close is None or benchmark_close.empty:
        return math.nan
    joined = pd.concat(
        [close.pct_change(), benchmark_close.pct_change()], axis=1, join="inner"
    ).dropna()
    joined = joined.iloc[-window:]
    if len(joined) < 60:
        return math.nan
    stock, market = joined.iloc[:, 0], joined.iloc[:, 1]
    variance = market.var(ddof=1)
    if variance == 0:
        return math.nan
    return float(stock.cov(market) / variance)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    returns = returns.dropna()
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return math.nan
    excess = returns - risk_free / TRADING_DAYS
    return float(excess.mean() / returns.std(ddof=1) * math.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series) -> float:
    returns = returns.dropna()
    downside = returns[returns < 0]
    if len(downside) < 2 or downside.std(ddof=1) == 0:
        return math.nan
    return float(returns.mean() / downside.std(ddof=1) * math.sqrt(TRADING_DAYS))


def cagr(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return math.nan
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return math.nan
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def performance_stats(returns: pd.Series) -> dict[str, float]:
    """Summary statistics of a daily return series."""
    returns = returns.fillna(0.0)
    equity = (1 + returns).cumprod()
    return {
        "total_return": float(equity.iloc[-1] - 1) if len(equity) else math.nan,
        "cagr": cagr(equity),
        "volatility": float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
        if len(returns) > 1
        else math.nan,
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": float(drawdown_series(equity).min()) if len(equity) else math.nan,
    }


def price_metrics(ind: pd.DataFrame, benchmark_close: pd.Series | None) -> dict[str, float]:
    """Snapshot metrics for the screener table (``ind`` from ``add_indicators``)."""
    close = ind["close"]
    last = ind.iloc[-1]
    returns_1y = close.pct_change().dropna().iloc[-TRADING_DAYS:]
    high_52w = last.get("high_52w", np.nan)
    return {
        "price": float(last["close"]),
        "ret_1m": period_return(close, 21),
        "ret_3m": period_return(close, 63),
        "ret_6m": period_return(close, 126),
        "ret_12m": period_return(close, 252),
        "mom_12_1": float(last.get("mom_12_1", np.nan)),
        "volatility": annualized_volatility(close),
        "max_drawdown": max_drawdown(close),
        "beta": beta(close, benchmark_close),
        "sharpe_1y": sharpe_ratio(returns_1y),
        "dist_52w_high": float(last["close"] / high_52w - 1)
        if high_52w and not math.isnan(high_52w)
        else math.nan,
        "rsi14": float(last.get("rsi14", np.nan)),
        "atr14": float(last.get("atr14", np.nan)),
        "sma50": float(last.get("sma50", np.nan)),
        "sma200": float(last.get("sma200", np.nan)),
    }
