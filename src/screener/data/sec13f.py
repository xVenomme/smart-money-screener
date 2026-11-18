"""SEC EDGAR client and Form 13F-HR parser.

Institutional managers with more than $100M in US equities must file a
Form 13F-HR every quarter, within 45 days of quarter end. The filing lists
their long positions in US-listed securities (by CUSIP).

Pipeline
--------
1. ``data.sec.gov/submissions/CIK##########.json`` -> list of filings
2. ``/Archives/edgar/data/<cik>/<accession>/index.json`` -> filing folder
3. the XML "information table" inside the folder -> positions

Gotchas handled here
--------------------
* Values should be in dollars for filings after 2023-01-03, but several
  managers still report in thousands: the unit is detected from the
  implied share price instead of trusting the filing date.
* The same CUSIP can appear on several rows (different sub-managers or
  voting authority): rows are summed.
* Options (``putCall``) and principal amounts (bonds, ``PRN``) are removed.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import requests

from screener.cache import DiskCache

log = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}"
FOREVER = float("inf")


@dataclass(frozen=True)
class Filing:
    cik: int
    accession: str  # e.g. 0001067983-25-000002
    form: str
    filing_date: date
    report_date: date | None

    @property
    def folder(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, accession=self.accession.replace("-", ""))


class SecClient:
    """Polite EDGAR client: identifies itself, throttles and caches."""

    def __init__(self, user_agent: str, cache: DiskCache, min_interval: float = 0.15) -> None:
        if "@" not in user_agent:
            log.warning("SEC asks for a contact e-mail in the User-Agent (set SEC_USER_AGENT)")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.cache = cache
        self.min_interval = min_interval
        self._last_call = 0.0

    def _get(self, url: str) -> requests.Response:
        for attempt in range(4):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            response = self.session.get(url, timeout=30)
            if response.status_code in (429, 500, 502, 503):
                time.sleep(2 ** (attempt + 1))
                continue
            response.raise_for_status()
            return response
        response.raise_for_status()
        return response

    def list_13f_filings(self, cik: int) -> tuple[str, list[Filing]]:
        """Entity name and its 13F-HR filings, newest first."""
        key = f"sec:submissions:{cik}"
        payload = self.cache.get(key, ttl_hours=12)
        if payload is None:
            payload = self._get(SUBMISSIONS_URL.format(cik=cik)).json()
            self.cache.set(key, payload)

        recent = payload.get("filings", {}).get("recent", {})
        filings = []
        for acc, form, filed, period in zip(
            recent.get("accessionNumber", []),
            recent.get("form", []),
            recent.get("filingDate", []),
            recent.get("reportDate", []),
            strict=False,
        ):
            if form != "13F-HR":  # amendments (13F-HR/A) are ignored on purpose
                continue
            filings.append(
                Filing(
                    cik=cik,
                    accession=acc,
                    form=form,
                    filing_date=date.fromisoformat(filed),
                    report_date=date.fromisoformat(period) if period else None,
                )
            )
        filings.sort(key=lambda f: (f.report_date or f.filing_date, f.filing_date), reverse=True)
        # one filing per reporting period (keep the most recent submission)
        unique: dict[date | None, Filing] = {}
        for filing in filings:
            unique.setdefault(filing.report_date, filing)
        return payload.get("name", ""), list(unique.values())

    def information_table(self, filing: Filing) -> bytes:
        """Raw XML of the filing's information table (cached forever)."""
        key = f"sec:infotable:{filing.accession}"
        cached = self.cache.get(key, ttl_hours=FOREVER)
        if cached is not None:
            return cached

        index = self._get(f"{filing.folder}/index.json").json()
        names = [item["name"] for item in index.get("directory", {}).get("item", [])]
        candidates = [
            n for n in names if n.lower().endswith(".xml") and n.lower() != "primary_doc.xml"
        ]
        candidates.sort(key=lambda n: ("info" not in n.lower() and "table" not in n.lower(), n))
        for name in candidates:
            content = self._get(f"{filing.folder}/{name}").content
            if b"infotable" in content[:5000].lower():
                self.cache.set(key, content)
                return content
        raise LookupError(f"No information table found in {filing.folder}")


# ---------------------------------------------------------------- parsing
def _local(tag: str) -> str:
    """Tag name without namespace (``{ns}infoTable`` / ``ns1:infoTable``)."""
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _find_text(element: ET.Element, *path: str) -> str | None:
    node = element
    for name in path:
        node = next((c for c in node if _local(c.tag).lower() == name.lower()), None)
        if node is None:
            return None
    return (node.text or "").strip()


def _to_float(text: str | None) -> float:
    if not text:
        return float("nan")
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return float("nan")


def parse_information_table(xml: bytes | str) -> pd.DataFrame:
    """Parse a 13F information table into one row per ``infoTable`` entry."""
    root = ET.fromstring(xml)
    rows = []
    for entry in root.iter():
        if _local(entry.tag) != "infoTable":
            continue
        put_call = _find_text(entry, "putCall")
        rows.append(
            {
                "issuer": _find_text(entry, "nameOfIssuer") or "",
                "title_class": _find_text(entry, "titleOfClass") or "",
                "cusip": (_find_text(entry, "cusip") or "").upper(),
                "value": _to_float(_find_text(entry, "value")),
                "shares": _to_float(_find_text(entry, "shrsOrPrnAmt", "sshPrnamt")),
                "share_type": (_find_text(entry, "shrsOrPrnAmt", "sshPrnamtType") or "SH").upper(),
                "put_call": put_call.upper() if put_call else None,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["issuer", "title_class", "cusip", "value", "shares", "share_type", "put_call"],
    )


def normalize_value_units(table: pd.DataFrame) -> pd.DataFrame:
    """Convert values to dollars when a filer reported them in thousands.

    The median implied price (value / shares) of common-stock rows is a few
    dollars or more when values are in dollars, and well below $1 when they
    are in thousands.
    """
    out = table.copy()
    stock = out[(out["share_type"] == "SH") & out["put_call"].isna() & (out["shares"] > 0)]
    if stock.empty:
        return out
    implied_price = float(np.nanmedian(stock["value"] / stock["shares"]))
    if implied_price < 1.0:
        out["value"] = out["value"] * 1000.0
        out.attrs["value_unit"] = "thousands"
    else:
        out.attrs["value_unit"] = "dollars"
    return out


def aggregate_positions(table: pd.DataFrame) -> pd.DataFrame:
    """One row per CUSIP (common stock only) with portfolio weight."""
    stock = table[(table["share_type"] == "SH") & table["put_call"].isna()]
    if stock.empty:
        return pd.DataFrame(columns=["issuer", "title_class", "value", "shares", "weight"])
    grouped = stock.groupby("cusip").agg(
        issuer=("issuer", "first"),
        title_class=("title_class", "first"),
        value=("value", "sum"),
        shares=("shares", "sum"),
    )
    total = grouped["value"].sum()
    grouped["weight"] = grouped["value"] / total if total > 0 else 0.0
    return grouped.sort_values("value", ascending=False)


def load_positions(client: SecClient, filing: Filing) -> pd.DataFrame:
    table = parse_information_table(client.information_table(filing))
    return aggregate_positions(normalize_value_units(table))
