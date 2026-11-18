"""Transparent, points-based risk classification: Low / Medium / High.

Each metric earns 0, 1 or 2 risk points against fixed thresholds (see
``config.yaml``). Fixed thresholds - instead of ranking stocks against
each other - keep the label meaningful in absolute terms: in a calm market
most stocks can be "Low", in a crash most become "High".

Missing data earns 1 point (unknown is not the same as safe).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

RISK_LEVELS = ("Low", "Medium", "High")


@dataclass(frozen=True)
class RiskResult:
    level: str
    points: int
    max_points: int
    breakdown: dict[str, int] = field(default_factory=dict)


def _missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def points_higher_is_riskier(value: float | None, low_max: float, medium_max: float) -> int:
    if _missing(value):
        return 1
    if value <= low_max:
        return 0
    if value <= medium_max:
        return 1
    return 2


def points_lower_is_riskier(value: float | None, safe_min: float, medium_min: float) -> int:
    if _missing(value):
        return 1
    if value >= safe_min:
        return 0
    if value >= medium_min:
        return 1
    return 2


def classify_risk(
    volatility: float | None,
    beta: float | None,
    max_drawdown: float | None,
    debt_to_equity: float | None,
    market_cap: float | None,
    cfg: dict,
) -> RiskResult:
    """Score a stock's risk. ``max_drawdown`` may be negative (it is abs()-ed)."""
    vol_weight = int(cfg.get("volatility_weight", 1))
    drawdown = None if _missing(max_drawdown) else abs(max_drawdown)

    def th(key: str) -> tuple[float, float]:  # YAML may hand us strings like "50e9"
        low, high = cfg[key]
        return float(low), float(high)

    breakdown = {
        "volatility": points_higher_is_riskier(volatility, *th("volatility")) * vol_weight,
        "beta": points_higher_is_riskier(beta, *th("beta")),
        "max_drawdown": points_higher_is_riskier(drawdown, *th("max_drawdown")),
        "debt_to_equity": points_higher_is_riskier(debt_to_equity, *th("debt_to_equity")),
        "market_cap": points_lower_is_riskier(market_cap, *th("market_cap")),
    }
    total = sum(breakdown.values())
    max_points = 2 * vol_weight + 8

    if total <= cfg["buckets"]["low"]:
        level = "Low"
    elif total <= cfg["buckets"]["medium"]:
        level = "Medium"
    else:
        level = "High"

    extreme = cfg.get("volatility_extreme")
    if extreme is not None and not _missing(volatility) and volatility > extreme:
        level = "High"

    return RiskResult(level, total, max_points, breakdown)
