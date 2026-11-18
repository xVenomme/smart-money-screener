import pandas as pd
import pytest

from screener.data.sec13f import aggregate_positions, normalize_value_units, parse_information_table
from screener.smart_money import (
    FundSnapshot,
    FundSpec,
    build_consensus,
    compare_quarters,
    demo_snapshots,
    load_funds,
    most_bought,
)
from tests.conftest import FIXTURES


def positions(rows: dict[str, tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(rows, orient="index", columns=["issuer", "value", "shares"])


def test_parse_dollar_table_aggregates_and_filters():
    table = parse_information_table((FIXTURES / "infotable_dollars.xml").read_bytes())
    assert len(table) == 5
    assert table.loc[table["cusip"] == "78462F103", "put_call"].item() == "PUT"

    normalized = normalize_value_units(table)
    assert normalized.attrs["value_unit"] == "dollars"

    pos = aggregate_positions(normalized)
    assert list(pos.index) == ["037833100", "025816109"]  # option + bond removed
    assert pos.loc["037833100", "shares"] == 250_000_000  # two rows summed
    assert pos["weight"].sum() == pytest.approx(1.0)


def test_thousands_are_detected_and_prefixed_namespace_parsed():
    table = parse_information_table((FIXTURES / "infotable_thousands.xml").read_bytes())
    assert table["value"].iloc[0] == 892_310  # "892,310" parsed
    normalized = normalize_value_units(table)
    assert normalized.attrs["value_unit"] == "thousands"
    assert normalized["value"].iloc[0] == 892_310_000


def test_compare_quarters_labels():
    prev = positions(
        {"A": ("A", 100, 100), "B": ("B", 100, 100), "C": ("C", 100, 100), "D": ("D", 100, 100)}
    )
    cur = positions(
        {"A": ("A", 150, 150), "B": ("B", 50, 50), "C": ("C", 101, 101), "E": ("E", 10, 10)}
    )
    out = compare_quarters(cur, prev)
    assert out["change"].to_dict() == {
        "A": "ADD",
        "B": "REDUCE",
        "C": "HOLD",
        "D": "SOLD",
        "E": "NEW",
    }
    assert out.loc["A", "share_change"] == pytest.approx(0.5)


def test_consensus_counts_funds():
    f1 = FundSnapshot(
        FundSpec("Fund 1", "x"),
        positions({"A": ("A", 10, 1), "B": ("B", 10, 1)}),
        positions({"A": ("A", 10, 1)}),
        tickers={"A": "AAA", "B": "BBB"},
    )
    f2 = FundSnapshot(
        FundSpec("Fund 2", "y"),
        positions({"A": ("A", 30, 3)}),
        positions({"A": ("A", 10, 1), "B": ("B", 5, 1)}),
        tickers={"A": "AAA", "B": "BBB"},
    )
    consensus = build_consensus([f1, f2])
    a, b = consensus.loc["A"], consensus.loc["B"]
    assert a["holders"] == 2 and a["added"] == 1 and a["net_buyers"] == 1
    assert b["holders"] == 1 and b["new"] == 1 and b["sold"] == 1 and b["net_buyers"] == 0
    assert a["funds"] == "Fund 1, Fund 2"
    assert most_bought(consensus).index[0] == "A"


def test_bundled_funds_file_is_valid():
    funds = load_funds()
    assert len(funds) >= 10
    assert len({f.cik for f in funds}) == len(funds)


def test_demo_smart_money_maps_to_tickers():
    result = demo_snapshots()
    holders = result.holders_by_ticker()
    assert not result.consensus.empty
    assert holders.index.str.endswith(".D").all()
