"""Provider interface shared by the live (Yahoo) and offline (demo) sources."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

# Normalised fundamental fields every provider must return (NaN when unknown).
FUNDAMENTAL_FIELDS: tuple[str, ...] = (
    "name",
    "sector",
    "industry",
    "currency",
    "market_cap",
    "forward_pe",
    "trailing_pe",
    "price_to_book",
    "ev_to_ebitda",
    "fcf_yield",  # free cash flow / market cap
    "roe",  # return on equity (fraction)
    "profit_margin",  # net margin (fraction)
    "debt_to_equity",  # percent, Yahoo convention (150 = 1.5x)
    "revenue_growth",  # yoy, fraction
    "earnings_growth",  # yoy, fraction
    "dividend_yield",  # fraction, computed from dividend rate / price
    "payout_ratio",  # fraction
    "target_price",  # analysts' mean target
    "analyst_rating",  # 1 = strong buy ... 5 = strong sell
)

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class MarketDataProvider(Protocol):
    """Anything that can return daily prices and a fundamentals snapshot."""

    name: str

    def get_history(self, tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
        """Daily OHLCV per ticker, columns ``open high low close volume``.

        Prices must be adjusted for splits and dividends. Tickers that could
        not be downloaded are simply missing from the result.
        """
        ...

    def get_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        """One row per ticker (index), columns = ``FUNDAMENTAL_FIELDS``."""
        ...


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names/index and drop rows without a close."""
    out = df.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    if "adj_close" in out.columns and "close" not in out.columns:
        out = out.rename(columns={"adj_close": "close"})
    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    for col in missing:
        out[col] = out["close"] if col != "volume" else 0.0
    out = out[list(OHLCV_COLUMNS)].astype(float)
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["close"])
