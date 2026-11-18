import math

import numpy as np
import pandas as pd
import pytest

from screener.analysis.risk import classify_risk
from screener.analysis.scoring import compute_scores, percentile_score


def test_defensive_stock_is_low_risk(config):
    r = classify_risk(0.15, 0.5, -0.10, 80, 150e9, config["risk"])
    assert r.level == "Low" and r.points == 0


def test_speculative_stock_is_high_risk(config):
    r = classify_risk(0.40, 1.6, -0.50, 250, 2e9, config["risk"])
    assert r.level == "High"
    assert r.points == r.max_points


def test_extreme_volatility_forces_high(config):
    r = classify_risk(0.60, 0.5, -0.10, 10, 500e9, config["risk"])
    assert r.level == "High"


def test_missing_data_counts_as_medium_points(config):
    r = classify_risk(None, math.nan, None, None, None, config["risk"])
    assert r.breakdown == {
        "volatility": 2,
        "beta": 1,
        "max_drawdown": 1,
        "debt_to_equity": 1,
        "market_cap": 1,
    }
    assert r.level == "Medium"


def test_percentile_direction_and_missing():
    s = pd.Series([10.0, 20.0, 30.0, np.nan])
    assert list(percentile_score(s)[:3]) == [0, 50, 100]
    assert list(percentile_score(s, higher_is_better=False)[:3]) == [100, 50, 0]
    assert np.isnan(percentile_score(s).iloc[3])


def test_negative_pe_is_worst():
    pe = pd.Series([-5.0, 10.0, 30.0])
    scores = percentile_score(pe, higher_is_better=False, non_positive_is_worst=True)
    assert scores.iloc[0] == 0 and scores.iloc[1] == 100


def test_income_profile_prefers_dividends_and_flags_yield_traps(config):
    table = pd.DataFrame(
        {
            "dividend_yield": [0.0, 0.03, 0.06, 0.10],
            "payout_ratio": [0.0, 0.5, 0.7, 1.5],  # last one pays more than it earns
            "roe": [0.15] * 4,
            "profit_margin": [0.1] * 4,
        },
        index=["NODIV", "MID", "HIGH", "TRAP"],
    )
    scored, weights = compute_scores(table, config["profiles"]["income"])
    assert scored["opportunity_score"].idxmax() == "HIGH"
    assert scored.loc["TRAP", "income_score"] == pytest.approx(50.0)  # 100 halved
    assert "smart_money" not in weights  # no data -> dropped
    assert sum(weights.values()) == pytest.approx(1.0)
