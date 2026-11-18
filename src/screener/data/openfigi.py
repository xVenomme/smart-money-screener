"""CUSIP -> ticker mapping through the free OpenFIGI API.

13F filings identify securities by CUSIP, while market data providers use
tickers. OpenFIGI (Bloomberg) offers a free mapping endpoint:
* without API key: 25 requests/minute, 10 identifiers per request
* with a free key (env ``OPENFIGI_API_KEY``): 25 requests/6s, 100 per request

Results (including "not found") are cached forever.
"""

from __future__ import annotations

import logging
import time

import requests

from screener.cache import DiskCache

log = logging.getLogger(__name__)

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
_CACHE_KEY = "openfigi:cusip-map"


def _pick_ticker(data: list[dict]) -> str | None:
    if not data:
        return None
    equities = [d for d in data if d.get("marketSector") == "Equity"] or data
    ticker = equities[0].get("ticker")
    if not ticker:
        return None
    return ticker.replace("/", "-").replace(" ", "-").upper()


class CusipMapper:
    def __init__(self, cache: DiskCache, api_key: str | None = None) -> None:
        self.cache = cache
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"
        if api_key:
            self.session.headers["X-OPENFIGI-APIKEY"] = api_key

    def map(self, cusips: list[str]) -> dict[str, str | None]:
        mapping: dict[str, str | None] = self.cache.get(_CACHE_KEY, ttl_hours=float("inf")) or {}
        todo = [c for c in dict.fromkeys(cusips) if c and c not in mapping]
        batch = 100 if self.api_key else 10
        pause = 0.3 if self.api_key else 2.6

        for start in range(0, len(todo), batch):
            chunk = todo[start : start + batch]
            jobs = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in chunk]
            try:
                response = self.session.post(OPENFIGI_URL, json=jobs, timeout=30)
                if response.status_code == 429:
                    log.info("OpenFIGI rate limit reached, waiting 60s")
                    time.sleep(60)
                    response = self.session.post(OPENFIGI_URL, json=jobs, timeout=30)
                response.raise_for_status()
            except requests.RequestException as exc:
                log.warning("OpenFIGI request failed: %s", exc)
                break
            for cusip, item in zip(chunk, response.json(), strict=False):
                mapping[cusip] = _pick_ticker(item.get("data", []))
            self.cache.set(_CACHE_KEY, mapping)
            time.sleep(pause)

        return {c: mapping.get(c) for c in cusips}
