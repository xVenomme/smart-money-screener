"""Offline demo data: fictional companies with synthetic but realistic behaviour.

Used by ``--demo`` / the dashboard demo mode and by the test-suite, so the
whole pipeline runs without network access or API limits. Every series is
generated from a seeded RNG, so results are reproducible.

Tickers end in ``.D`` to make clear they are not real securities.
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from screener.data.base import FUNDAMENTAL_FIELDS

TRADING_DAYS = 252
DEMO_DAYS = 1300  # ~5 years of history, enough for the backtest
DEMO_BENCHMARK = "MKT.D"


@dataclass(frozen=True)
class Archetype:
    mu: float  # long-run annual drift
    sigma: float  # annual volatility
    beta: float
    pe: tuple[float, float]
    div: tuple[float, float]
    payout: tuple[float, float]
    de: tuple[float, float]
    roe: tuple[float, float]
    margin: tuple[float, float]
    rev_g: tuple[float, float]
    eps_g: tuple[float, float]
    mcap: tuple[float, float]


ARCHETYPES: dict[str, Archetype] = {
    "defensive": Archetype(
        0.07,
        0.15,
        0.55,
        (14, 20),
        (0.025, 0.045),
        (0.45, 0.70),
        (60, 160),
        (0.10, 0.18),
        (0.08, 0.15),
        (0.00, 0.05),
        (0.00, 0.07),
        (40e9, 180e9),
    ),
    "income": Archetype(
        0.05,
        0.17,
        0.70,
        (15, 22),
        (0.050, 0.070),
        (0.70, 0.90),
        (90, 160),
        (0.08, 0.14),
        (0.10, 0.20),
        (0.00, 0.04),
        (0.00, 0.05),
        (15e9, 60e9),
    ),
    "quality": Archetype(
        0.13,
        0.22,
        0.95,
        (24, 32),
        (0.005, 0.015),
        (0.15, 0.35),
        (20, 80),
        (0.25, 0.45),
        (0.20, 0.35),
        (0.08, 0.15),
        (0.10, 0.18),
        (100e9, 900e9),
    ),
    "growth": Archetype(
        0.18,
        0.36,
        1.35,
        (40, 70),
        (0.0, 0.0),
        (0.0, 0.0),
        (0, 50),
        (0.12, 0.30),
        (0.05, 0.20),
        (0.20, 0.45),
        (0.25, 0.60),
        (20e9, 300e9),
    ),
    "cyclical": Archetype(
        0.06,
        0.28,
        1.20,
        (8, 13),
        (0.020, 0.040),
        (0.25, 0.45),
        (80, 180),
        (0.08, 0.16),
        (0.05, 0.10),
        (-0.05, 0.08),
        (-0.10, 0.10),
        (10e9, 60e9),
    ),
    "speculative": Archetype(
        0.04,
        0.60,
        1.70,
        (-40, -10),
        (0.0, 0.0),
        (0.0, 0.0),
        (100, 300),
        (-0.30, 0.00),
        (-0.40, -0.05),
        (0.10, 0.60),
        (-0.50, 0.20),
        (0.8e9, 5e9),
    ),
    "value_trap": Archetype(
        -0.08,
        0.30,
        1.00,
        (6, 9),
        (0.070, 0.110),
        (1.10, 1.60),
        (150, 300),
        (0.02, 0.08),
        (0.01, 0.05),
        (-0.12, -0.02),
        (-0.30, -0.05),
        (5e9, 25e9),
    ),
}

DEMO_COMPANIES: list[tuple[str, str, str, str]] = [
    ("AQUA.D", "Aquaterra Utilities", "Utilities", "defensive"),
    ("GRID.D", "Gridline Power", "Utilities", "defensive"),
    ("HRTH.D", "Hearthstone Foods", "Consumer Defensive", "defensive"),
    ("PILL.D", "Pellucid Pharma", "Healthcare", "defensive"),
    ("INSR.D", "Insurex Group", "Financial Services", "defensive"),
    ("FOOD.D", "Freshfield Grocers", "Consumer Defensive", "income"),
    ("TOLL.D", "Tollway Infrastructure", "Industrials", "income"),
    ("REIT.D", "Brickvault REIT", "Real Estate", "income"),
    ("SOFT.D", "Softwave Systems", "Technology", "quality"),
    ("NOVA.D", "Novaris Medical", "Healthcare", "quality"),
    ("PAY.D", "Paystream Networks", "Financial Services", "quality"),
    ("LUXE.D", "Lumiere Luxury", "Consumer Cyclical", "quality"),
    ("RAIL.D", "Railbridge Logistics", "Industrials", "quality"),
    ("AERO.D", "Aerolux Defense", "Industrials", "quality"),
    ("CHIP.D", "Chipforge Semiconductors", "Technology", "growth"),
    ("CLDX.D", "Cloudex Platforms", "Technology", "growth"),
    ("ROBO.D", "Robotica Automation", "Industrials", "growth"),
    ("GAME.D", "Pixelforge Games", "Communication Services", "growth"),
    ("DATA.D", "Datanomics AI", "Technology", "growth"),
    ("MINE.D", "Ironvale Mining", "Basic Materials", "cyclical"),
    ("OILX.D", "Petrox Energy", "Energy", "cyclical"),
    ("STEL.D", "Forgeworks Steel", "Basic Materials", "cyclical"),
    ("AUTO.D", "Autoline Motors", "Consumer Cyclical", "cyclical"),
    ("BANK.D", "Bancorpia", "Financial Services", "cyclical"),
    ("SHIP.D", "Seaborne Shipping", "Industrials", "cyclical"),
    ("GENE.D", "Genexa Biotech", "Healthcare", "speculative"),
    ("VOLT.D", "Voltaic EV", "Consumer Cyclical", "speculative"),
    ("QBIT.D", "Qubitron Quantum", "Technology", "speculative"),
    ("TELE.D", "Telemark Comms", "Communication Services", "value_trap"),
    ("MALL.D", "Mallgrove Retail", "Consumer Cyclical", "value_trap"),
]

DEMO_TICKERS = [c[0] for c in DEMO_COMPANIES]
_BY_TICKER = {c[0]: c for c in DEMO_COMPANIES}

MARKET_MU, MARKET_SIGMA = 0.08, 0.16


def _rng(*keys: str | int) -> np.random.Generator:
    seed = zlib.crc32("|".join(map(str, keys)).encode()) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def _regime_drift(rng: np.random.Generator, n: int, base_mu: float, spread: float) -> np.ndarray:
    """Piecewise-constant annual drift: creates trends, corrections and rallies."""
    drift = np.empty(n)
    i = 0
    while i < n:
        length = int(rng.integers(60, 260))
        drift[i : i + length] = rng.normal(base_mu, spread)
        i += length
    return drift / TRADING_DAYS


def _dates(n: int) -> pd.DatetimeIndex:
    end = pd.Timestamp.today().normalize()
    if end.dayofweek >= 5:
        end -= pd.offsets.BDay(1)
    return pd.bdate_range(end=end, periods=n)


def _market_returns() -> np.ndarray:
    rng = _rng("market")
    drift = _regime_drift(rng, DEMO_DAYS, MARKET_MU, 0.15)
    shocks = rng.standard_t(df=5, size=DEMO_DAYS) / math.sqrt(5 / 3)
    return drift + shocks * MARKET_SIGMA / math.sqrt(TRADING_DAYS)


def _ohlcv_from_returns(
    rng: np.random.Generator, log_ret: np.ndarray, start: float, sigma: float
) -> pd.DataFrame:
    close = start * np.exp(np.cumsum(log_ret))
    prev_close = np.r_[start, close[:-1]]
    daily_sigma = sigma / math.sqrt(TRADING_DAYS)
    open_ = prev_close * np.exp(rng.normal(0, daily_sigma * 0.3, len(close)))
    top = np.maximum(open_, close)
    bottom = np.minimum(open_, close)
    high = top * (1 + np.abs(rng.normal(0, daily_sigma * 0.5, len(close))))
    low = bottom * (1 - np.abs(rng.normal(0, daily_sigma * 0.5, len(close))))
    volume = rng.lognormal(mean=14, sigma=0.4, size=len(close)).round()
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=_dates(len(close)),
    )


def _period_to_days(period: str) -> int:
    match = re.fullmatch(r"(\d+)(d|mo|y)", period)
    if period == "max" or not match:
        return DEMO_DAYS
    n, unit = int(match.group(1)), match.group(2)
    days = {"d": n, "mo": n * 21, "y": n * TRADING_DAYS}[unit]
    return min(days, DEMO_DAYS)


class DemoProvider:
    """Synthetic market: a regime-switching index plus 30 fictional stocks."""

    name = "demo"

    def __init__(self) -> None:
        self._market = _market_returns()
        self._cache: dict[str, pd.DataFrame] = {}

    def _series(self, ticker: str) -> pd.DataFrame:
        if ticker not in self._cache:
            self._cache[ticker] = self._generate(ticker)
        return self._cache[ticker]

    def _generate(self, ticker: str) -> pd.DataFrame:
        if ticker == DEMO_BENCHMARK:
            rng = _rng("market-ohlc")
            log_ret = np.log1p(self._market)
            return _ohlcv_from_returns(rng, log_ret, 4000.0, MARKET_SIGMA)
        if ticker not in _BY_TICKER:
            raise KeyError(ticker)
        archetype = ARCHETYPES[_BY_TICKER[ticker][3]]
        rng = _rng(ticker)
        idio_sigma = math.sqrt(
            max(archetype.sigma**2 - (archetype.beta * MARKET_SIGMA) ** 2, 0.05**2)
        )
        idio_drift = _regime_drift(rng, DEMO_DAYS, archetype.mu - archetype.beta * MARKET_MU, 0.15)
        idio = (
            rng.standard_t(df=4, size=DEMO_DAYS)
            / math.sqrt(2)
            * idio_sigma
            / math.sqrt(TRADING_DAYS)
        )
        simple = archetype.beta * self._market + idio_drift + idio
        log_ret = np.log1p(np.clip(simple, -0.5, 0.5))
        start = float(rng.uniform(15, 150))
        return _ohlcv_from_returns(rng, log_ret, start, archetype.sigma)

    def get_history(self, tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
        days = _period_to_days(period)
        out = {}
        for ticker in tickers:
            try:
                out[ticker] = self._series(ticker).iloc[-days:]
            except KeyError:
                continue
        return out

    def get_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        rows = {}
        for ticker in tickers:
            if ticker not in _BY_TICKER:
                continue
            _, name, sector, kind = _BY_TICKER[ticker]
            a = ARCHETYPES[kind]
            rng = _rng(ticker, "fundamentals")

            def u(bounds: tuple[float, float], rng: np.random.Generator = rng) -> float:
                return float(rng.uniform(*bounds))

            last_close = float(self._series(ticker)["close"].iloc[-1])
            pe = u(a.pe)
            market_cap = u(a.mcap)
            rows[ticker] = {
                "name": name,
                "sector": sector,
                "industry": f"{sector} (demo)",
                "currency": "USD",
                "market_cap": market_cap,
                "forward_pe": pe * float(rng.uniform(0.85, 0.95)) if pe > 0 else pe,
                "trailing_pe": pe if pe > 0 else math.nan,
                "price_to_book": max(0.4, pe / 5 * float(rng.uniform(0.6, 1.4)))
                if pe > 0
                else float(rng.uniform(2, 8)),
                "ev_to_ebitda": max(3.0, pe * 0.6 * float(rng.uniform(0.8, 1.2)))
                if pe > 0
                else math.nan,
                "fcf_yield": (1 / pe) * float(rng.uniform(0.6, 1.2))
                if pe > 0
                else float(rng.uniform(-0.15, -0.02)),
                "roe": u(a.roe),
                "profit_margin": u(a.margin),
                "debt_to_equity": u(a.de),
                "revenue_growth": u(a.rev_g),
                "earnings_growth": u(a.eps_g),
                "dividend_yield": u(a.div),
                "payout_ratio": u(a.payout),
                "target_price": last_close * (1 + float(rng.normal(0.08, 0.12))),
                "analyst_rating": float(np.clip(rng.normal(2.3, 0.5), 1.0, 4.5)),
            }
        frame = pd.DataFrame.from_dict(rows, orient="index")
        return frame.reindex(columns=list(FUNDAMENTAL_FIELDS))


# ----------------------------------------------------------------- demo 13F
DEMO_FUNDS: list[tuple[str, str, list[str]]] = [
    (
        "Demo Value Partners",
        "Fictional value investor",
        ["defensive", "income", "cyclical", "value_trap"],
    ),
    ("Demo Quality Compounders", "Fictional quality investor", ["quality", "defensive"]),
    ("Demo Growth Capital", "Fictional growth investor", ["growth", "quality", "speculative"]),
    ("Demo Macro Fund", "Fictional macro investor", ["cyclical", "growth", "quality"]),
    ("Demo Income Trust", "Fictional income investor", ["income", "defensive"]),
    ("Demo Tiger Cub", "Fictional long/short fund", ["growth", "quality", "cyclical"]),
]


def demo_holdings() -> list[dict]:
    """Two quarters of fictional 13F-style holdings for the demo funds.

    Returns a list of dicts with keys ``name``, ``manager``, ``current`` and
    ``previous`` (DataFrames indexed by CUSIP with issuer/value/shares).
    """
    provider = DemoProvider()
    prices = {t: float(provider._series(t)["close"].iloc[-1]) for t in DEMO_TICKERS}
    funds = []
    for fund_name, manager, styles in DEMO_FUNDS:
        rng = _rng(fund_name)
        pool = [c for c in DEMO_COMPANIES if c[3] in styles]
        size = min(len(pool), int(rng.integers(6, 11)))
        picks = list(rng.choice(len(pool), size=size, replace=False))
        prev_rows, cur_rows = {}, {}
        for idx in picks:
            ticker, name, _, _ = pool[idx]
            cusip = f"DEMO{ticker.split('.')[0]:0<5}"
            shares = float(rng.integers(50, 500)) * 10_000
            state = rng.choice(
                ["hold", "add", "reduce", "new", "sold"], p=[0.35, 0.25, 0.15, 0.15, 0.10]
            )
            prev_shares = {
                "hold": shares,
                "add": shares * 0.7,
                "reduce": shares * 1.5,
                "new": 0.0,
                "sold": shares,
            }[state]
            cur_shares = 0.0 if state == "sold" else shares
            row = {"issuer": name.upper(), "ticker": ticker}
            if prev_shares:
                prev_rows[cusip] = row | {
                    "shares": prev_shares,
                    "value": prev_shares * prices[ticker] * 0.95,
                }
            if cur_shares:
                cur_rows[cusip] = row | {"shares": cur_shares, "value": cur_shares * prices[ticker]}
        funds.append(
            {
                "name": fund_name,
                "manager": manager,
                "current": pd.DataFrame.from_dict(cur_rows, orient="index"),
                "previous": pd.DataFrame.from_dict(prev_rows, orient="index"),
            }
        )
    return funds
