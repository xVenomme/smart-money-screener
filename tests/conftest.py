from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from screener.config import load_config
from screener.data.demo import DemoProvider

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def demo_provider() -> DemoProvider:
    return DemoProvider()


def make_ohlcv(close: np.ndarray | list[float], start: str = "2020-01-01") -> pd.DataFrame:
    """OHLCV frame from a close series (open = previous close, +-1% range)."""
    close = np.asarray(close, dtype=float)
    index = pd.bdate_range(start, periods=len(close))
    prev = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": prev,
            "high": np.maximum(prev, close) * 1.01,
            "low": np.minimum(prev, close) * 0.99,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=index,
    )


@pytest.fixture
def uptrend() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    returns = 0.0015 + rng.normal(0, 0.008, 600)
    return make_ohlcv(100 * np.exp(np.cumsum(returns)))


@pytest.fixture
def downtrend() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    returns = -0.0015 + rng.normal(0, 0.008, 600)
    return make_ohlcv(100 * np.exp(np.cumsum(returns)))
