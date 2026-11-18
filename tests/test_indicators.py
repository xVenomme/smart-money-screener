import numpy as np
import pandas as pd
import pytest

from screener.analysis.indicators import add_indicators, atr, macd, rsi, sma
from tests.conftest import make_ohlcv


def test_sma_matches_manual_mean():
    s = pd.Series([1.0, 2, 3, 4, 5])
    out = sma(s, 3)
    assert out.isna().sum() == 2
    assert out.iloc[-1] == pytest.approx(4.0)


def test_rsi_extremes_and_flat():
    up = pd.Series(np.arange(1, 60, dtype=float))
    down = up[::-1].reset_index(drop=True)
    flat = pd.Series(np.full(60, 10.0))
    assert rsi(up).iloc[-1] == pytest.approx(100.0)
    assert rsi(down).iloc[-1] == pytest.approx(0.0)
    assert rsi(flat).iloc[-1] == pytest.approx(50.0)


def test_rsi_alternating_is_neutral():
    s = pd.Series(100 + np.tile([1.0, -1.0], 100).cumsum())  # 101, 100, 101, 100...
    assert 40 < rsi(s).iloc[-1] < 60


def test_macd_sign_follows_trend():
    up = pd.Series(100 * 1.002 ** np.arange(300))
    m = macd(up)
    assert m["macd"].iloc[-1] > 0
    assert macd(up[::-1].reset_index(drop=True))["macd"].iloc[-1] < 0
    assert list(m.columns) == ["macd", "macd_signal", "macd_hist"]


def test_atr_of_constant_range():
    close = np.full(50, 100.0)
    df = make_ohlcv(close)
    df["high"], df["low"] = 102.0, 98.0
    assert atr(df["high"], df["low"], df["close"]).iloc[-1] == pytest.approx(4.0)


def test_indicators_have_no_lookahead(uptrend):
    """Values up to day t must not change when future data is appended."""
    full = add_indicators(uptrend)
    partial = add_indicators(uptrend.iloc[:400])
    pd.testing.assert_frame_equal(full.iloc[:400], partial, check_freq=False)
