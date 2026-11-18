"""Factor-based opportunity score (0-100).

Each metric is converted to a cross-sectional percentile score inside the
screened universe (0 = worst, 100 = best). Metrics are averaged into
factors, factors are combined with profile weights (``config.yaml``) and
the composite is ranked again, so the opportunity score reads as
"better than X% of the stocks screened". It is a *relative* score.

Factors
-------
value        cheap vs earnings, EBITDA, book value; high free-cash-flow yield
quality      high ROE and margins, low leverage
growth       revenue and earnings growth
momentum     12-1 month and 6 month price momentum
income       dividend yield (halved when payout ratio > 100%: "yield trap")
upside       analysts' mean target vs current price
smart_money  number of tracked superinvestors holding / buying the stock
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# factor -> [(metric, higher_is_better, non_positive_is_worst)]
FACTORS: dict[str, list[tuple[str, bool, bool]]] = {
    "value": [
        ("forward_pe", False, True),  # negative P/E = losses = worst
        ("ev_to_ebitda", False, True),
        ("price_to_book", False, True),
        ("fcf_yield", True, False),
    ],
    "quality": [
        ("roe", True, False),
        ("profit_margin", True, False),
        ("debt_to_equity", False, False),
    ],
    "growth": [
        ("revenue_growth", True, False),
        ("earnings_growth", True, False),
    ],
    "momentum": [
        ("mom_12_1", True, False),
        ("ret_6m", True, False),
    ],
    "income": [("dividend_yield", True, False)],
    "upside": [("analyst_upside", True, False)],
    "smart_money": [
        ("smart_money_holders", True, False),
        ("smart_money_net_buyers", True, False),
    ],
}


def percentile_score(
    series: pd.Series, higher_is_better: bool = True, non_positive_is_worst: bool = False
) -> pd.Series:
    """Rank-based score in [0, 100]; NaN stays NaN (handled by the caller)."""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if non_positive_is_worst:
        worst = -np.inf if higher_is_better else np.inf
        values = values.where(values.isna() | (values > 0), worst)
    valid = values.notna().sum()
    if valid == 0:
        return pd.Series(np.nan, index=series.index)
    if valid == 1:
        return pd.Series(np.where(values.notna(), 50.0, np.nan), index=series.index)
    ranks = values.rank(ascending=higher_is_better, method="average")
    return (ranks - 1) / (valid - 1) * 100


def compute_scores(
    table: pd.DataFrame, weights: dict[str, float]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Add ``<factor>_score`` columns and ``opportunity_score``.

    Returns the enriched table and the weights actually used (factors with
    no data at all are dropped and the rest re-normalised).
    """
    out = table.copy()
    available: dict[str, bool] = {}
    for factor, metrics in FACTORS.items():
        parts = [
            percentile_score(out[metric], higher, non_pos)
            for metric, higher, non_pos in metrics
            if metric in out.columns
        ]
        if parts:
            score = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
        else:
            score = pd.Series(np.nan, index=out.index)
        if factor == "income" and "payout_ratio" in out.columns:
            yield_trap = pd.to_numeric(out["payout_ratio"], errors="coerce") > 1.0
            score = score.where(~yield_trap, score * 0.5)
        available[factor] = bool(score.notna().any())
        out[f"{factor}_score"] = score.round(1)

    used = {f: float(w) for f, w in weights.items() if f in FACTORS and available.get(f) and w > 0}
    total = sum(used.values())
    if total == 0:
        out["opportunity_score"] = 50.0
        return out, {}
    used = {f: w / total for f, w in used.items()}
    composite = sum(out[f"{f}_score"].fillna(50.0) * w for f, w in used.items())
    out["composite_raw"] = composite.round(1)
    # Averaging several percentiles compresses values towards 50, so the
    # final score is re-ranked: 80 = better than 80% of the universe.
    out["opportunity_score"] = percentile_score(composite).round(1)
    return out, used
