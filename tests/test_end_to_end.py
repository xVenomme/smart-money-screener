import pytest

from screener.cli import main
from screener.data.universe import load_universe
from screener.screener import filter_table, run_screen
from screener.smart_money import demo_snapshots


def test_demo_screen_end_to_end(config, demo_provider):
    holders = demo_snapshots().holders_by_ticker()
    result = run_screen("demo", demo_provider, config, "balanced", holders)
    table = result.table
    assert len(table) == 30
    assert table["opportunity_score"].between(0, 100).all()
    assert set(table["risk"]) <= {"Low", "Medium", "High"}
    assert table["opportunity_score"].is_monotonic_decreasing
    assert table["smart_money_holders"].notna().all()
    buys = table[table["action"].isin(["STRONG BUY", "BUY"])]
    assert (buys["stop_loss"] < buys["entry"]).all()
    low_risk = filter_table(table, risk=["Low"])
    assert (low_risk["risk"] == "Low").all()


@pytest.mark.parametrize("profile", ["balanced", "income", "growth", "value", "smart_money"])
def test_every_profile_runs(config, demo_provider, profile):
    result = run_screen("demo", demo_provider, config, profile)
    assert abs(sum(result.weights.values()) - 1) < 1e-9


def test_unknown_universe_and_profile(config, demo_provider):
    with pytest.raises(ValueError):
        load_universe("does-not-exist", config)
    with pytest.raises(ValueError):
        run_screen("demo", demo_provider, config, "yolo")


def test_bundled_universes_load(config):
    us = load_universe("us_large_caps", config)
    it = load_universe("italy_ftse_mib", config)
    assert len(us.tickers) >= 90 and us.benchmark == "SPY"
    assert all(t.endswith(".MI") for t in it.tickers)


@pytest.mark.parametrize(
    "argv",
    [
        ["scan", "--demo", "--top", "5"],
        ["scan", "--demo", "--risk", "Low", "--smart-money"],
        ["smart-money", "--demo", "--top", "5"],
        ["plan", "CLDX.D", "--demo"],
        ["backtest", "AQUA.D", "--demo", "--years", "3"],
        ["universes"],
    ],
)
def test_cli_commands(argv, capsys):
    assert main(argv) == 0
