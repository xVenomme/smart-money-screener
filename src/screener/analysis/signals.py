"""When to buy, when to sell.

Two independent ingredients:

1. **Technical score** (-100 ... +100): *timing*. Five simple votes, each
   -1 / 0 / +1, computed every day from prices only - so it can be
   backtested without look-ahead bias:

   ============  =========================================  ===============
   component     bullish (+1)                               bearish (-1)
   ============  =========================================  ===============
   trend         close > SMA200                             close < SMA200
   regime        SMA50 > SMA200 ("golden cross")            SMA50 < SMA200
   macd          MACD line > signal line                    MACD < signal
   rsi           RSI < 30 (oversold, 0 otherwise)           RSI > 70
   momentum      12-1 month return > 0                      < 0
   ============  =========================================  ===============

2. **Opportunity score** (0 ... 100): *what* to buy (see ``scoring.py``).

The action combines both, and for buy actions a trade plan adds an entry
price, an ATR-based stop-loss (wider for riskier stocks), a take-profit at
a fixed reward/risk ratio and explicit exit rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BUY_ACTIONS = ("STRONG BUY", "BUY", "BUY ON PULLBACK")
ACTIONS = ("STRONG BUY", "BUY", "BUY ON PULLBACK", "WATCH", "HOLD", "SELL", "STRONG SELL", "N/A")
COMPONENTS = ("trend", "regime", "macd", "rsi", "momentum")


def technical_components(ind: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Daily votes (-1/0/+1) of each component; NaN while indicators warm up."""

    def vote(series: pd.Series) -> pd.Series:
        return np.sign(series).where(series.notna())

    rsi = ind["rsi14"]
    rsi_vote = pd.Series(0.0, index=ind.index)
    rsi_vote[rsi < cfg["rsi_oversold"]] = 1.0
    rsi_vote[rsi > cfg["rsi_overbought"]] = -1.0
    return pd.DataFrame(
        {
            "trend": vote(ind["close"] - ind["sma200"]),
            "regime": vote(ind["sma50"] - ind["sma200"]),
            "macd": vote(ind["macd"] - ind["macd_signal"]),
            "rsi": rsi_vote.where(rsi.notna()),
            "momentum": vote(ind["mom_12_1"]),
        },
        index=ind.index,
    )


def technical_score(ind: pd.DataFrame, cfg: dict) -> pd.Series:
    """Mean of available votes x 100. Requires the long-term trend (SMA200)."""
    comps = technical_components(ind, cfg)
    score = comps.mean(axis=1, skipna=True) * 100
    return score.where(comps["trend"].notna() & comps["regime"].notna()).round(0)


def decide_action(tech: float, opportunity: float, rsi: float, cfg: dict) -> str:
    """Combine timing (technical score) and quality (opportunity score)."""
    if tech is None or math.isnan(tech):
        return "N/A"
    rsi = 50.0 if rsi is None or math.isnan(rsi) else rsi
    if tech <= -cfg["strong_threshold"] and opportunity < 40:
        return "STRONG SELL"
    if tech <= cfg["sell_threshold"]:
        return "SELL"
    if (
        tech >= cfg["strong_threshold"]
        and opportunity >= cfg["min_opportunity_strong"]
        and rsi <= cfg["rsi_overbought"]
    ):
        return "STRONG BUY"
    if tech >= cfg["buy_threshold"] and opportunity >= cfg["min_opportunity_buy"]:
        return "BUY ON PULLBACK" if rsi > cfg["rsi_overbought"] else "BUY"
    if opportunity >= cfg["min_opportunity_watch"]:
        return "WATCH"
    return "HOLD"


@dataclass
class TradePlan:
    action: str
    technical_score: float
    opportunity_score: float
    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    analyst_target: float | None = None
    risk_pct: float | None = None
    reward_pct: float | None = None
    reasons: list[str] = field(default_factory=list)
    exit_rules: list[str] = field(default_factory=list)


def _reasons(last: pd.Series, comps: pd.Series) -> list[str]:
    out = []
    labels = {
        "trend": (
            "Price above its 200-day average: long-term uptrend",
            "Price below its 200-day average: long-term downtrend",
        ),
        "regime": (
            "50-day average above 200-day (golden-cross regime)",
            "50-day average below 200-day (death-cross regime)",
        ),
        "macd": (
            "MACD above its signal line: momentum improving",
            "MACD below its signal line: momentum fading",
        ),
    }
    for key, (bull, bear) in labels.items():
        value = comps.get(key)
        if value is not None and not math.isnan(value):
            out.append(("+ " + bull) if value > 0 else ("- " + bear))
    rsi = last.get("rsi14", math.nan)
    if not math.isnan(rsi):
        if comps.get("rsi", 0) > 0:
            out.append(f"+ RSI {rsi:.0f}: oversold, potential rebound")
        elif comps.get("rsi", 0) < 0:
            out.append(f"- RSI {rsi:.0f}: overbought, stretched short term")
        else:
            out.append(f"= RSI {rsi:.0f}: neutral")
    mom = last.get("mom_12_1", math.nan)
    if not math.isnan(mom):
        out.append(f"{'+' if mom > 0 else '-'} 12-1 month momentum {mom:+.1%}")
    return out


def build_trade_plan(
    ind: pd.DataFrame,
    opportunity: float,
    risk_level: str,
    analyst_target: float | None,
    cfg: dict,
) -> TradePlan:
    """Action + concrete levels for the latest bar of ``ind``."""
    comps = technical_components(ind, cfg)
    tech_series = technical_score(ind, cfg)
    last = ind.iloc[-1]
    tech = float(tech_series.iloc[-1])
    close = float(last["close"])
    atr = float(last.get("atr14", math.nan))
    rsi = float(last.get("rsi14", math.nan))
    sma20, sma50, sma200 = (float(last.get(c, math.nan)) for c in ("sma20", "sma50", "sma200"))

    action = decide_action(tech, opportunity, rsi, cfg)
    target = analyst_target if analyst_target and not math.isnan(analyst_target) else None
    plan = TradePlan(
        action, tech, opportunity, analyst_target=target, reasons=_reasons(last, comps.iloc[-1])
    )

    k_stop = cfg["atr_stop_multiplier"].get(risk_level, 2.5)
    k_trail = cfg["trailing_atr_multiplier"]
    reward_risk = cfg["reward_risk"]

    if action in BUY_ACTIONS and not math.isnan(atr) and atr > 0:
        entry = close
        if action == "BUY ON PULLBACK":
            pullback = close - atr
            entry = min(close, max(sma20, pullback) if not math.isnan(sma20) else pullback)
        stop = entry - k_stop * atr
        take_profit = entry + reward_risk * (entry - stop)
        plan.entry, plan.stop_loss, plan.take_profit = (
            round(entry, 2),
            round(stop, 2),
            round(take_profit, 2),
        )
        plan.risk_pct = (stop / entry) - 1
        plan.reward_pct = (take_profit / entry) - 1
        when = (
            "now (at market)"
            if action != "BUY ON PULLBACK"
            else f"on a pullback, limit order near {entry:.2f}"
        )
        plan.exit_rules = [
            f"Entry: {when}",
            f"Stop-loss: {stop:.2f} ({k_stop:g} x ATR below entry, {plan.risk_pct:+.1%})",
            f"Take profit: {take_profit:.2f} (reward/risk {reward_risk:g}:1, {plan.reward_pct:+.1%})",
            f"Trailing stop: highest close since entry - {k_trail:g} x ATR",
            f"Sell if the technical score drops to {cfg['sell_threshold']} or below",
        ]
        if not math.isnan(sma200):
            plan.exit_rules.append(
                f"Long-term exit: close below the 200-day average ({sma200:.2f})"
            )
    elif action in ("SELL", "STRONG SELL"):
        plan.exit_rules = ["Exit existing positions / avoid new ones"]
        if not math.isnan(sma200):
            plan.exit_rules.append(
                f"Re-evaluate when price closes back above the 200-day average ({sma200:.2f})"
            )
    elif action in ("WATCH", "HOLD"):
        plan.exit_rules = [
            f"Buy trigger: technical score >= {cfg['buy_threshold']} (now {tech:+.0f})"
        ]
        if not math.isnan(sma50):
            plan.exit_rules.append(
                f"Typical trigger: close above the 50-day average ({sma50:.2f}) with MACD turning up"
            )
        if action == "HOLD":
            plan.exit_rules.append(
                f"If already owned: keep while price stays above the 200-day average ({sma200:.2f})"
            )
    return plan
