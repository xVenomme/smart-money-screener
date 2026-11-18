"""Live market data from Yahoo Finance through the ``yfinance`` library.

Notes
-----
* ``yfinance`` is an unofficial client: Yahoo can rate limit or change the
  payload at any time, so every call is wrapped with retries and every
  result is cached on disk.
* Dividend yield is computed as ``dividendRate / price`` instead of reading
  ``dividendYield``, whose unit (fraction vs percent) changed across
  yfinance versions.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from screener.cache import DiskCache
from screener.data.base import FUNDAMENTAL_FIELDS, clean_ohlcv

log = logging.getLogger(__name__)

_CHUNK = 50


def _num(value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def normalize_info(info: dict) -> dict:
    """Map a raw ``yfinance`` ``.info`` dict onto ``FUNDAMENTAL_FIELDS``."""
    price = _num(
        info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    )
    market_cap = _num(info.get("marketCap"))
    fcf = _num(info.get("freeCashflow"))
    dividend_rate = _num(info.get("dividendRate") or info.get("trailingAnnualDividendRate"))

    dividend_yield = 0.0
    if not math.isnan(dividend_rate) and price and not math.isnan(price) and price > 0:
        dividend_yield = dividend_rate / price

    return {
        "name": info.get("shortName") or info.get("longName") or "",
        "sector": info.get("sector") or "Unknown",
        "industry": info.get("industry") or "Unknown",
        "currency": info.get("currency") or "",
        "market_cap": market_cap,
        "forward_pe": _num(info.get("forwardPE")),
        "trailing_pe": _num(info.get("trailingPE")),
        "price_to_book": _num(info.get("priceToBook")),
        "ev_to_ebitda": _num(info.get("enterpriseToEbitda")),
        "fcf_yield": fcf / market_cap
        if market_cap and not math.isnan(fcf) and market_cap > 0
        else math.nan,
        "roe": _num(info.get("returnOnEquity")),
        "profit_margin": _num(info.get("profitMargins")),
        "debt_to_equity": _num(info.get("debtToEquity")),
        "revenue_growth": _num(info.get("revenueGrowth")),
        "earnings_growth": _num(info.get("earningsGrowth")),
        "dividend_yield": dividend_yield,
        "payout_ratio": _num(info.get("payoutRatio")),
        "target_price": _num(info.get("targetMeanPrice")),
        "analyst_rating": _num(info.get("recommendationMean")),
    }


def split_download(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Split a multi-ticker ``yf.download`` frame into one frame per ticker."""
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))
        for ticker in tickers:
            if ticker in level0:
                frame = raw[ticker]
            elif ticker in level1:
                frame = raw.xs(ticker, axis=1, level=1)
            else:
                continue
            frame = frame.dropna(how="all")
            if not frame.empty and {"Close", "close"} & set(map(str, frame.columns)):
                cleaned = clean_ohlcv(frame)
                if not cleaned.empty:
                    out[ticker] = cleaned
    elif len(tickers) == 1:
        cleaned = clean_ohlcv(raw.dropna(how="all"))
        if not cleaned.empty:
            out[tickers[0]] = cleaned
    return out


class YahooProvider:
    name = "yahoo"

    def __init__(self, cache: DiskCache, max_workers: int = 8, retries: int = 3) -> None:
        self.cache = cache
        self.max_workers = max_workers
        self.retries = retries

    # ------------------------------------------------------------------ prices
    def get_history(self, tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
        import yfinance as yf

        result: dict[str, pd.DataFrame] = {}
        to_download: list[str] = []
        for ticker in dict.fromkeys(tickers):
            cached = self.cache.get(f"hist:{period}:{ticker}")
            if cached is not None:
                result[ticker] = cached
            else:
                to_download.append(ticker)

        for start in range(0, len(to_download), _CHUNK):
            chunk = to_download[start : start + _CHUNK]
            raw = None
            for attempt in range(self.retries):
                try:
                    raw = yf.download(
                        chunk,
                        period=period,
                        interval="1d",
                        auto_adjust=True,
                        group_by="ticker",
                        threads=True,
                        progress=False,
                    )
                    break
                except Exception as exc:  # network / rate limit
                    wait = 2 ** (attempt + 1)
                    log.warning("yf.download failed (%s), retrying in %ss", exc, wait)
                    time.sleep(wait)
            for ticker, frame in split_download(raw, chunk).items():
                self.cache.set(f"hist:{period}:{ticker}", frame)
                result[ticker] = frame
        return result

    # ------------------------------------------------------------ fundamentals
    def _fetch_info(self, ticker: str) -> dict:
        import yfinance as yf

        cached = self.cache.get(f"info:{ticker}")
        if cached is not None:
            return cached
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                info = yf.Ticker(ticker).info or {}
                row = normalize_info(info)
                self.cache.set(f"info:{ticker}", row)
                return row
            except Exception as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        log.warning("Fundamentals unavailable for %s: %s", ticker, last_exc)
        return {field: math.nan for field in FUNDAMENTAL_FIELDS} | {
            "name": "",
            "sector": "Unknown",
            "industry": "Unknown",
            "currency": "",
            "dividend_yield": 0.0,
        }

    def get_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        rows: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._fetch_info, t): t for t in dict.fromkeys(tickers)}
            for future in as_completed(futures):
                rows[futures[future]] = future.result()
        frame = pd.DataFrame.from_dict(rows, orient="index")
        return frame.reindex(columns=list(FUNDAMENTAL_FIELDS))
