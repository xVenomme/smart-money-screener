"""Backtest of the timing rules on a single stock, against buy & hold.

Execution model (deliberately conservative):
* the signal is computed on the **close** of day ``t`` using data up to ``t``;
* orders are filled at the **open** of day ``t+1`` (no trading on the same
  close that generated the signal);
* every entry and exit pays ``cost_bps`` (commission + slippage);
* long only, fully invested or flat.

Rules
-----
* enter when technical score >= ``entry_threshold``
* exit when technical score <= ``exit_threshold`` (hysteresis avoids
  whipsaws) or when the close falls below a chandelier trailing stop:
  highest close since entry - ``k`` x ATR.

Only prices are used. Fundamentals are *not* backtested because free
sources only give today's values: using them in the past would be
look-ahead bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from screener.analysis.indicators import add_indicators
from screener.analysis.metrics import performance_stats
from screener.analysis.signals import technical_score


@dataclass
class BacktestResult:
    equity: pd.Series  # strategy equity curve (starts at 1)
    buy_hold: pd.Series  # buy & hold equity curve (starts at 1)
    position: pd.Series  # 1 when invested during the day, else 0
    score: pd.Series
    trades: pd.DataFrame
    stats: pd.DataFrame  # rows: metrics, columns: Strategy / Buy & Hold


def desired_positions(
    close: np.ndarray,
    score: np.ndarray,
    atr: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    trail_k: float,
) -> np.ndarray:
    """Position decided at each close (1 = long from the next open).

    Only ``close[:t+1]``, ``score[:t+1]`` and ``atr[:t+1]`` are read at step t.
    """
    n = len(close)
    desired = np.zeros(n)
    in_position = False
    highest = -math.inf
    for t in range(n):
        s, c, a = score[t], close[t], atr[t]
        if in_position:
            highest = max(highest, c)
            stop = highest - trail_k * a if not math.isnan(a) else -math.inf
            if (not math.isnan(s) and s <= exit_threshold) or c < stop:
                in_position = False
        elif not math.isnan(s) and s >= entry_threshold:
            in_position = True
            highest = c
        desired[t] = 1.0 if in_position else 0.0
    return desired


def run_backtest(ohlcv: pd.DataFrame, signal_cfg: dict, bt_cfg: dict) -> BacktestResult:
    ind = add_indicators(ohlcv)
    score = technical_score(ind, signal_cfg)

    # Start once all indicators are warmed up (fair comparison with buy & hold).
    valid = score.notna() & ind["mom_12_1"].notna()
    if valid.sum() < 30:
        raise ValueError("Not enough history for a backtest (need ~1.5 years of daily data)")
    start = valid.idxmax()
    ind, score = ind.loc[start:], score.loc[start:]

    o, c = ind["open"].to_numpy(), ind["close"].to_numpy()
    desired = desired_positions(
        c,
        score.to_numpy(),
        ind["atr14"].to_numpy(),
        bt_cfg["entry_threshold"],
        bt_cfg["exit_threshold"],
        bt_cfg["trailing_atr_multiplier"],
    )
    cost = bt_cfg["cost_bps"] / 10_000

    held = np.r_[0.0, desired[:-1]]  # position during day t (decided at close t-1)
    held_before = np.r_[0.0, held[:-1]]  # position during day t-1
    prev_c = np.r_[c[0], c[:-1]]
    entering = (held == 1) & (held_before == 0)
    exiting = (held == 0) & (held_before == 1)
    staying = (held == 1) & (held_before == 1)
    daily = np.zeros(len(c))
    daily[staying] = c[staying] / prev_c[staying] - 1
    daily[entering] = c[entering] / o[entering] - 1 - cost  # bought at the open
    daily[exiting] = o[exiting] / prev_c[exiting] - 1 - cost  # sold at the open

    index = ind.index
    strategy_returns = pd.Series(daily, index=index)
    bh_returns = pd.Series(c / prev_c - 1, index=index)
    bh_returns.iloc[0] = 0.0

    trades = _trades(index, o, c, held, cost)
    stats = pd.DataFrame(
        {
            "Strategy": performance_stats(strategy_returns),
            "Buy & Hold": performance_stats(bh_returns),
        }
    )
    exposure = float(held.mean())
    closed = trades[trades["status"] == "closed"]
    extra = pd.DataFrame(
        {
            "Strategy": {
                "exposure": exposure,
                "trades": float(len(trades)),
                "win_rate": float((closed["return"] > 0).mean()) if len(closed) else math.nan,
                "avg_trade": float(closed["return"].mean()) if len(closed) else math.nan,
            },
            "Buy & Hold": {
                "exposure": 1.0,
                "trades": 1.0,
                "win_rate": math.nan,
                "avg_trade": math.nan,
            },
        }
    )
    return BacktestResult(
        equity=(1 + strategy_returns).cumprod(),
        buy_hold=(1 + bh_returns).cumprod(),
        position=pd.Series(held, index=index),
        score=score,
        trades=trades,
        stats=pd.concat([stats, extra]),
    )


def _trades(
    index: pd.DatetimeIndex, o: np.ndarray, c: np.ndarray, held: np.ndarray, cost: float
) -> pd.DataFrame:
    rows = []
    entry_i = None
    for t in range(len(held)):
        prev = held[t - 1] if t else 0.0
        if held[t] == 1 and prev == 0:
            entry_i = t
        elif held[t] == 0 and prev == 1 and entry_i is not None:
            rows.append(_trade_row(index, entry_i, t, o[entry_i], o[t], "closed", cost))
            entry_i = None
    if entry_i is not None:
        rows.append(_trade_row(index, entry_i, len(held) - 1, o[entry_i], c[-1], "open", cost))
    return pd.DataFrame(
        rows,
        columns=[
            "entry_date",
            "entry_price",
            "exit_date",
            "exit_price",
            "return",
            "days",
            "status",
        ],
    )


def _trade_row(index, i, j, entry_price, exit_price, status, cost) -> dict:
    """Net return: both legs pay transaction costs."""
    return {
        "entry_date": index[i].date(),
        "entry_price": round(float(entry_price), 2),
        "exit_date": index[j].date(),
        "exit_price": round(float(exit_price), 2),
        "return": float(exit_price / entry_price - 1 - 2 * cost),
        "days": int((index[j] - index[i]).days),
        "status": status,
    }
