"""Orchestrator: universe -> data -> metrics -> scores -> risk -> signals."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from screener.analysis.indicators import add_indicators
from screener.analysis.metrics import price_metrics
from screener.analysis.risk import classify_risk
from screener.analysis.scoring import compute_scores
from screener.analysis.signals import TradePlan, build_trade_plan
from screener.data.base import MarketDataProvider
from screener.data.universe import Universe, load_universe

log = logging.getLogger(__name__)

MIN_HISTORY_DAYS = 210  # SMA200 + a few days

TABLE_COLUMNS = [
    "name",
    "sector",
    "price",
    "action",
    "opportunity_score",
    "technical_score",
    "risk",
    "risk_points",
    "entry",
    "stop_loss",
    "take_profit",
    "analyst_upside",
    "dividend_yield",
    "forward_pe",
    "roe",
    "revenue_growth",
    "ret_1m",
    "ret_6m",
    "ret_12m",
    "volatility",
    "beta",
    "max_drawdown",
    "rsi14",
    "smart_money_holders",
    "smart_money_net_buyers",
    "value_score",
    "quality_score",
    "growth_score",
    "momentum_score",
    "income_score",
    "upside_score",
    "smart_money_score",
    "market_cap",
    "debt_to_equity",
    "payout_ratio",
]

ProgressFn = Callable[[float, str], None]


@dataclass
class ScreenResult:
    table: pd.DataFrame
    indicators: dict[str, pd.DataFrame]
    plans: dict[str, TradePlan]
    universe: Universe
    profile: str
    weights: dict[str, float]
    provider: str
    benchmark: pd.Series | None = None
    generated_at: datetime = field(default_factory=datetime.now)
    warnings: list[str] = field(default_factory=list)


def run_screen(
    universe: str | Universe,
    provider: MarketDataProvider,
    config: dict,
    profile: str = "balanced",
    smart_money: pd.DataFrame | None = None,
    progress: ProgressFn | None = None,
) -> ScreenResult:
    """Screen a universe and return one row per stock, best opportunities first.

    ``smart_money`` is the optional output of
    ``SmartMoneyResult.holders_by_ticker()``.
    """
    report = progress or (lambda *_: None)
    uni = universe if isinstance(universe, Universe) else load_universe(universe, config)
    if profile not in config["profiles"]:
        raise ValueError(f"Unknown profile {profile!r}. Available: {', '.join(config['profiles'])}")
    warnings: list[str] = []

    report(0.05, f"Downloading prices for {len(uni.tickers)} stocks")
    history = provider.get_history(uni.tickers + [uni.benchmark], config["data"]["history_period"])
    bench = history.get(uni.benchmark)
    bench_close = bench["close"] if bench is not None else None
    if bench_close is None:
        warnings.append(f"Benchmark {uni.benchmark} unavailable: beta not computed")

    report(0.35, "Downloading fundamentals")
    fundamentals = provider.get_fundamentals(uni.tickers)

    report(0.70, "Computing indicators, risk and signals")
    rows: dict[str, dict] = {}
    indicators: dict[str, pd.DataFrame] = {}
    for ticker in uni.tickers:
        ohlcv = history.get(ticker)
        if ohlcv is None or len(ohlcv) < MIN_HISTORY_DAYS:
            warnings.append(f"{ticker}: not enough price history, skipped")
            continue
        ind = add_indicators(ohlcv)
        indicators[ticker] = ind
        row = price_metrics(ind, bench_close)
        if ticker in fundamentals.index:
            fund = fundamentals.loc[ticker].to_dict()
            row.update({k: v for k, v in fund.items() if k not in row})
        target = row.get("target_price", math.nan)
        row["analyst_upside"] = (
            target / row["price"] - 1 if target and not math.isnan(target) else math.nan
        )
        rows[ticker] = row

    if not rows:
        raise RuntimeError("No data downloaded - check the tickers or your internet connection")

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "ticker"
    for col in ("name", "sector"):
        if col not in table.columns:
            table[col] = ""
    table["name"] = table["name"].fillna("").astype(str)
    table["sector"] = table["sector"].fillna("Unknown").replace("", "Unknown")

    if smart_money is not None and not smart_money.empty:
        table = table.join(smart_money, how="left")
        table[["smart_money_holders", "smart_money_net_buyers"]] = table[
            ["smart_money_holders", "smart_money_net_buyers"]
        ].fillna(0)
    else:
        table["smart_money_holders"] = math.nan
        table["smart_money_net_buyers"] = math.nan

    table, weights = compute_scores(table, config["profiles"][profile])

    plans: dict[str, TradePlan] = {}
    risk_levels, risk_points = [], []
    for ticker, row in table.iterrows():
        risk = classify_risk(
            row.get("volatility"),
            row.get("beta"),
            row.get("max_drawdown"),
            row.get("debt_to_equity"),
            row.get("market_cap"),
            config["risk"],
        )
        risk_levels.append(risk.level)
        risk_points.append(risk.points)
        plans[ticker] = build_trade_plan(
            indicators[ticker],
            float(row["opportunity_score"]),
            risk.level,
            row.get("target_price"),
            config["signals"],
        )
    table["risk"] = risk_levels
    table["risk_points"] = risk_points
    table["action"] = [plans[t].action for t in table.index]
    table["technical_score"] = [plans[t].technical_score for t in table.index]
    for col, attr in (
        ("entry", "entry"),
        ("stop_loss", "stop_loss"),
        ("take_profit", "take_profit"),
    ):
        table[col] = pd.to_numeric(
            pd.Series([getattr(plans[t], attr) for t in table.index], index=table.index),
            errors="coerce",
        )

    table = table.reindex(columns=[c for c in TABLE_COLUMNS if c in table.columns])
    table = table.sort_values(["opportunity_score", "technical_score"], ascending=False)
    report(1.0, "Done")
    return ScreenResult(
        table=table,
        indicators=indicators,
        plans=plans,
        universe=uni,
        profile=profile,
        weights=weights,
        provider=getattr(provider, "name", "unknown"),
        benchmark=bench_close,
        warnings=warnings,
    )


def filter_table(
    table: pd.DataFrame,
    risk: list[str] | None = None,
    actions: list[str] | None = None,
    sectors: list[str] | None = None,
    min_opportunity: float = 0.0,
) -> pd.DataFrame:
    out = table
    if risk:
        out = out[out["risk"].isin(risk)]
    if actions:
        out = out[out["action"].isin(actions)]
    if sectors:
        out = out[out["sector"].isin(sectors)]
    return out[out["opportunity_score"] >= min_opportunity]
