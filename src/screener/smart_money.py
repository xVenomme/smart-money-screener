"""What are the big investors buying?

Aggregates the latest two 13F filings of a list of well-known managers
("superinvestors") into:
* a consensus table: how many funds hold each stock, total value, average
  portfolio weight, and how many funds opened / added / reduced / sold it;
* per-fund activity: the quarter-over-quarter changes of one manager.

Remember: 13F data is delayed (up to 45 days after quarter end) and shows
long positions only. It tells you what was held, not what is held today.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import yaml

from screener.config import resource_path

log = logging.getLogger(__name__)

CHANGE_LABELS = ("NEW", "ADD", "HOLD", "REDUCE", "SOLD")


@dataclass(frozen=True)
class FundSpec:
    name: str
    manager: str
    cik: int | None = None


@dataclass
class FundSnapshot:
    fund: FundSpec
    current: pd.DataFrame  # index cusip: issuer, value, shares, weight
    previous: pd.DataFrame | None = None
    report_date: date | None = None
    filing_date: date | None = None
    tickers: dict[str, str | None] = field(default_factory=dict)


@dataclass
class SmartMoneyResult:
    consensus: pd.DataFrame
    snapshots: list[FundSnapshot]
    warnings: list[str]
    source: str = "sec"

    def holders_by_ticker(self) -> pd.DataFrame:
        """Columns ``smart_money_holders`` and ``smart_money_net_buyers`` per ticker."""
        mapped = self.consensus.dropna(subset=["ticker"])
        if mapped.empty:
            return pd.DataFrame(columns=["smart_money_holders", "smart_money_net_buyers"])
        grouped = mapped.groupby("ticker").agg(
            smart_money_holders=("holders", "max"), smart_money_net_buyers=("net_buyers", "max")
        )
        return grouped


def load_funds(path: str | None = None) -> list[FundSpec]:
    file = path or resource_path("funds.yaml")
    with open(file, encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    return [FundSpec(f["name"], f.get("manager", ""), int(f["cik"])) for f in payload["funds"]]


# ---------------------------------------------------------------- analytics
def _with_weight(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "weight" not in out.columns:
        total = out["value"].sum()
        out["weight"] = out["value"] / total if total > 0 else 0.0
    return out


def compare_quarters(
    current: pd.DataFrame, previous: pd.DataFrame | None, tolerance: float = 0.02
) -> pd.DataFrame:
    """Classify each position as NEW / ADD / HOLD / REDUCE / SOLD.

    A change in share count within ``tolerance`` (2%) counts as HOLD.
    Note: stock splits change share counts too and would show up as ADD.
    """
    cols = ["issuer", "value", "shares", "weight"]
    cur = _with_weight(current)[cols] if not current.empty else pd.DataFrame(columns=cols)
    if previous is None:
        out = cur.copy()
        out["prev_shares"] = np.nan
        out["share_change"] = np.nan
        out["change"] = "n/a"
        return out

    prev = _with_weight(previous)[cols] if not previous.empty else pd.DataFrame(columns=cols)
    merged = cur.join(prev.add_prefix("prev_"), how="outer")
    merged["issuer"] = merged["issuer"].fillna(merged["prev_issuer"])
    shares = merged["shares"].astype(float).fillna(0.0)
    prev_shares = merged["prev_shares"].astype(float).fillna(0.0)

    merged["change"] = np.select(
        [
            (prev_shares == 0) & (shares > 0),
            (shares == 0) & (prev_shares > 0),
            shares > prev_shares * (1 + tolerance),
            shares < prev_shares * (1 - tolerance),
        ],
        ["NEW", "SOLD", "ADD", "REDUCE"],
        default="HOLD",
    )
    merged["share_change"] = np.where(
        prev_shares > 0, shares / prev_shares.replace(0, np.nan) - 1, np.nan
    )
    merged[["value", "shares", "weight"]] = (
        merged[["value", "shares", "weight"]].astype(float).fillna(0.0)
    )
    merged["prev_shares"] = prev_shares
    return merged[["issuer", "value", "shares", "weight", "prev_shares", "share_change", "change"]]


def fund_activity(snapshot: FundSnapshot) -> pd.DataFrame:
    """Positions of one fund with quarter-over-quarter change and ticker."""
    table = compare_quarters(snapshot.current, snapshot.previous)
    table.insert(0, "ticker", [snapshot.tickers.get(c) for c in table.index])
    return table.sort_values("value", ascending=False)  # exits (value 0) at the bottom


def build_consensus(snapshots: list[FundSnapshot]) -> pd.DataFrame:
    """Aggregate all funds into one row per security (CUSIP)."""
    records = []
    for snap in snapshots:
        table = compare_quarters(snap.current, snap.previous)
        for cusip, row in table.iterrows():
            records.append(
                {
                    "cusip": cusip,
                    "ticker": snap.tickers.get(cusip),
                    "issuer": row["issuer"],
                    "fund": snap.fund.name,
                    "value": row["value"],
                    "weight": row["weight"],
                    "change": row["change"],
                    "held": row["shares"] > 0,
                }
            )
    columns = [
        "ticker",
        "issuer",
        "holders",
        "total_value",
        "avg_weight",
        "max_weight",
        "new",
        "added",
        "reduced",
        "sold",
        "net_buyers",
        "funds",
    ]
    if not records:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(records)
    held = df[df["held"]]
    grouped = df.groupby("cusip")
    consensus = pd.DataFrame(
        {
            "ticker": grouped["ticker"].agg(lambda s: next((t for t in s if t), None)),
            "issuer": grouped["issuer"].first(),
            "holders": grouped["held"].sum().astype(int),
            "total_value": grouped["value"].sum(),
            "avg_weight": held.groupby("cusip")["weight"].mean(),
            "max_weight": held.groupby("cusip")["weight"].max(),
            "new": grouped["change"].agg(lambda s: int((s == "NEW").sum())),
            "added": grouped["change"].agg(lambda s: int((s == "ADD").sum())),
            "reduced": grouped["change"].agg(lambda s: int((s == "REDUCE").sum())),
            "sold": grouped["change"].agg(lambda s: int((s == "SOLD").sum())),
            "funds": held.groupby("cusip")["fund"].agg(lambda s: ", ".join(sorted(s))),
        }
    )
    consensus["avg_weight"] = consensus["avg_weight"].fillna(0.0)
    consensus["max_weight"] = consensus["max_weight"].fillna(0.0)
    consensus["funds"] = consensus["funds"].fillna("")
    consensus["net_buyers"] = (
        consensus["new"] + consensus["added"] - consensus["reduced"] - consensus["sold"]
    )
    consensus.index.name = "cusip"
    return consensus[columns].sort_values(["holders", "total_value"], ascending=False)


def most_bought(consensus: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    active = consensus[(consensus["new"] + consensus["added"]) > 0]
    return active.sort_values(["net_buyers", "new", "total_value"], ascending=False).head(top)


def most_sold(consensus: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    active = consensus[(consensus["reduced"] + consensus["sold"]) > 0]
    return active.sort_values(["net_buyers", "sold"], ascending=[True, False]).head(top)


# ------------------------------------------------------------------ loaders
def _cusips_to_map(snapshots: list[FundSnapshot], top_n: int) -> list[str]:
    """Only map what is shown: top positions of each fund + overlaps."""
    counts: dict[str, int] = {}
    selected: set[str] = set()
    for snap in snapshots:
        frames = [snap.current] + ([snap.previous] if snap.previous is not None else [])
        for frame in frames:
            for cusip in frame.index:
                counts[cusip] = counts.get(cusip, 0) + 1
        selected.update(snap.current.sort_values("value", ascending=False).index[:top_n])
    selected.update(c for c, n in counts.items() if n >= 3)  # held/traded by 2+ funds
    return sorted(selected)


def fetch_sec_snapshots(
    config: dict, funds: list[FundSpec] | None = None, progress=None
) -> SmartMoneyResult:
    """Download and analyse the latest 13F filings of the tracked funds."""
    from screener.cache import DiskCache
    from screener.data.openfigi import CusipMapper
    from screener.data.sec13f import SecClient, load_positions

    sm_cfg = config["smart_money"]
    cache = DiskCache(config["data"]["cache_dir"], config["data"]["cache_ttl_hours"])
    client = SecClient(sm_cfg["sec_user_agent"], cache)
    funds = funds or load_funds()
    snapshots: list[FundSnapshot] = []
    warnings: list[str] = []
    today = date.today()

    for i, fund in enumerate(funds):
        if progress:
            progress(i / max(len(funds), 1), f"13F: {fund.name}")
        try:
            entity, filings = client.list_13f_filings(fund.cik)
            if not filings:
                warnings.append(f"{fund.name}: no 13F-HR filings found (CIK {fund.cik})")
                continue
            first_word = fund.name.split()[0].lower()
            if entity and first_word not in entity.lower():
                warnings.append(
                    f"{fund.name}: CIK {fund.cik} belongs to '{entity}' - check funds.yaml"
                )
            latest = filings[0]
            age = (today - latest.filing_date).days
            if age > sm_cfg["stale_after_days"]:
                warnings.append(
                    f"{fund.name}: last 13F filed {latest.filing_date} ({age} days ago), skipped"
                )
                continue
            current = load_positions(client, latest)
            previous = load_positions(client, filings[1]) if len(filings) > 1 else None
            snapshots.append(
                FundSnapshot(fund, current, previous, latest.report_date, latest.filing_date)
            )
        except Exception as exc:  # one broken fund must not break the report
            log.exception("13F download failed for %s", fund.name)
            warnings.append(f"{fund.name}: download failed ({exc})")

    if progress:
        progress(0.95, "Mapping CUSIP -> ticker (OpenFIGI)")
    mapper = CusipMapper(cache, sm_cfg.get("openfigi_api_key"))
    mapping = mapper.map(_cusips_to_map(snapshots, sm_cfg["top_positions_to_map"]))
    for snap in snapshots:
        snap.tickers = mapping

    return SmartMoneyResult(build_consensus(snapshots), snapshots, warnings, source="sec")


def demo_snapshots() -> SmartMoneyResult:
    """Fictional funds holding the demo universe (offline mode)."""
    from screener.data.demo import demo_holdings

    snapshots = []
    quarter_end = (pd.Timestamp.today() - pd.offsets.QuarterEnd(1)).date()
    for fund in demo_holdings():
        tickers = {}
        for frame in (fund["current"], fund["previous"]):
            tickers.update(frame["ticker"].to_dict() if not frame.empty else {})
        snapshots.append(
            FundSnapshot(
                FundSpec(fund["name"], fund["manager"]),
                fund["current"].drop(columns="ticker"),
                fund["previous"].drop(columns="ticker"),
                report_date=quarter_end,
                filing_date=quarter_end + pd.Timedelta(days=45),
                tickers=tickers,
            )
        )
    return SmartMoneyResult(build_consensus(snapshots), snapshots, [], source="demo")


def run_smart_money(config: dict, demo: bool = False, progress=None) -> SmartMoneyResult:
    return demo_snapshots() if demo else fetch_sec_snapshots(config, progress=progress)
