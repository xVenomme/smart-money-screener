"""Ticker universes: bundled CSV lists, the S&P 500 from Wikipedia, or a custom CSV."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from screener.config import resource_path
from screener.data.demo import DEMO_BENCHMARK, DEMO_TICKERS

log = logging.getLogger(__name__)

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass(frozen=True)
class Universe:
    name: str
    tickers: list[str]
    benchmark: str
    description: str = ""


def _read_ticker_csv(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    column = "ticker" if "ticker" in frame.columns else frame.columns[0]
    tickers = frame[column].dropna().astype(str).str.strip().str.upper()
    return list(dict.fromkeys(t for t in tickers if t))


def _fetch_sp500() -> list[str]:
    import requests

    response = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0 (screener)"}, timeout=20)
    response.raise_for_status()
    table = pd.read_html(io.StringIO(response.text))[0]
    # Yahoo uses dashes for share classes: BRK.B -> BRK-B
    return [str(s).replace(".", "-").strip() for s in table["Symbol"]]


def available_universes(config: dict) -> dict[str, str]:
    names = {name: spec.get("description", "") for name, spec in config["universes"].items()}
    names["demo"] = "30 fictional companies with synthetic data (offline)"
    return names


def load_universe(name: str, config: dict) -> Universe:
    """Resolve a universe name (or a path to a CSV with a ``ticker`` column)."""
    if name == "demo":
        return Universe("demo", list(DEMO_TICKERS), DEMO_BENCHMARK, "Synthetic demo universe")

    path = Path(name)
    if path.suffix.lower() == ".csv" and path.exists():
        return Universe(path.stem, _read_ticker_csv(path), "SPY", f"Custom list ({path.name})")

    spec = config["universes"].get(name)
    if spec is None:
        raise ValueError(
            f"Unknown universe {name!r}. Available: {', '.join(available_universes(config))} "
            "or a path to a CSV file with a 'ticker' column."
        )

    if spec.get("source") == "wikipedia_sp500":
        try:
            tickers = _fetch_sp500()
        except Exception as exc:  # offline, layout change, HTTP error...
            log.warning(
                "Could not download the S&P 500 list (%s); using %s", exc, spec["fallback_file"]
            )
            tickers = _read_ticker_csv(resource_path("universes", spec["fallback_file"]))
    else:
        tickers = _read_ticker_csv(resource_path("universes", spec["file"]))

    return Universe(name, tickers, spec["benchmark"], spec.get("description", ""))
