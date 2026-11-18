"""Technical indicators (vectorised with pandas).

All indicators at row ``t`` use only data up to and including ``t``:
no centred windows, no backward filling. This is what makes the backtest
free of look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing (0-100)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    out = out.where(avg_loss != 0, 100.0)  # only gains
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)  # flat prices
    return out.where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (Wilder): typical daily range, used for stops."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def add_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame enriched with all indicators used downstream."""
    df = ohlcv.copy()
    close = df["close"]
    df["sma20"] = sma(close, 20)
    df["sma50"] = sma(close, 50)
    df["sma200"] = sma(close, 200)
    df["rsi14"] = rsi(close, 14)
    df = df.join(macd(close))
    df["atr14"] = atr(df["high"], df["low"], close, 14)
    # 12-1 momentum: return from 12 months ago to 1 month ago (skips the
    # most recent month, which tends to mean-revert - Jegadeesh & Titman).
    df["mom_12_1"] = close.shift(21) / close.shift(252) - 1
    df["high_52w"] = close.rolling(252, min_periods=20).max()
    return df.replace([np.inf, -np.inf], np.nan)
