"""Command line interface.

Examples
--------
    screener scan --demo                          # offline, synthetic data
    screener scan -u us_large_caps -p income --risk Low --top 15
    screener scan -u italy_ftse_mib --csv results.csv
    screener smart-money --top 25
    screener smart-money --fund "Berkshire Hathaway"
    screener plan AAPL -u us_large_caps
    screener backtest MSFT --years 5
"""

from __future__ import annotations

import argparse
import logging
import math
import sys

import pandas as pd
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from screener import __version__
from screener.config import load_config
from screener.data import get_provider
from screener.data.universe import available_universes

console = Console()

ACTION_STYLE = {
    "STRONG BUY": "bold green",
    "BUY": "green",
    "BUY ON PULLBACK": "yellow",
    "WATCH": "cyan",
    "HOLD": "white",
    "SELL": "red",
    "STRONG SELL": "bold red",
    "N/A": "dim",
}
RISK_STYLE = {"Low": "green", "Medium": "yellow", "High": "red"}
DISCLAIMER = "[dim]Educational project - not investment advice. Data may be delayed or wrong.[/dim]"


def _styled(text: str, style: str | None) -> str:
    text = escape(str(text))
    return f"[{style}]{text}[/]" if style else text


def _pct(x: float, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def _num(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:,.{digits}f}"


def _usd(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(x) >= scale:
            return f"${x / scale:,.1f}{unit}"
    return f"${x:,.0f}"


# ----------------------------------------------------------------- commands
def cmd_scan(args: argparse.Namespace, config: dict) -> int:
    from screener.screener import filter_table, run_screen

    provider = get_provider("demo" if args.demo else "yahoo", config)
    universe = "demo" if args.demo else args.universe
    holders = None
    if args.smart_money or args.profile == "smart_money":
        from screener.smart_money import run_smart_money

        with console.status("Downloading 13F filings..."):
            holders = run_smart_money(config, demo=args.demo).holders_by_ticker()

    with console.status(f"Screening {universe}...") as status:
        result = run_screen(
            universe,
            provider,
            config,
            args.profile,
            holders,
            progress=lambda _p, msg: status.update(msg),
        )

    table = filter_table(
        result.table, risk=args.risk, actions=args.action, min_opportunity=args.min_score
    )
    shown = table.head(args.top)

    out = Table(
        title=f"{result.universe.name} - profile '{args.profile}' - {len(table)}/{len(result.table)} stocks",
        header_style="bold",
    )
    for col in (
        "Ticker",
        "Name",
        "Action",
        "Opp.",
        "Tech.",
        "Risk",
        "Price",
        "Entry",
        "Stop",
        "Target",
        "Upside",
        "Yield",
        "12m",
        "Vol.",
        "Funds",
    ):
        out.add_column(
            col, justify="left" if col in ("Ticker", "Name", "Action", "Risk") else "right"
        )
    for ticker, row in shown.iterrows():
        holders_txt = (
            "-"
            if math.isnan(row.get("smart_money_holders", math.nan))
            else f"{row['smart_money_holders']:.0f}"
        )
        out.add_row(
            ticker,
            str(row["name"])[:22],
            _styled(row["action"], ACTION_STYLE.get(row["action"])),
            f"{row['opportunity_score']:.0f}",
            f"{row['technical_score']:+.0f}",
            _styled(row["risk"], RISK_STYLE.get(row["risk"])),
            _num(row["price"]),
            _num(row["entry"]),
            _num(row["stop_loss"]),
            _num(row["take_profit"]),
            _pct(row["analyst_upside"], True),
            _pct(row["dividend_yield"]),
            _pct(row["ret_12m"], True),
            _pct(row["volatility"]),
            holders_txt,
        )
    console.print(out)
    for warning in result.warnings[:10]:
        console.print(_styled(f"! {warning}", "yellow"))
    if args.csv:
        table.to_csv(args.csv)
        console.print(f"Saved {len(table)} rows to [bold]{args.csv}[/]")
    console.print(DISCLAIMER)
    return 0


def cmd_smart_money(args: argparse.Namespace, config: dict) -> int:
    from screener.smart_money import fund_activity, most_bought, most_sold, run_smart_money

    with console.status(
        "Downloading 13F filings (first run can take a couple of minutes)..."
    ) as status:
        result = run_smart_money(
            config, demo=args.demo, progress=lambda _p, msg: status.update(msg)
        )

    if args.fund:
        snap = next((s for s in result.snapshots if args.fund.lower() in s.fund.name.lower()), None)
        if snap is None:
            console.print(
                f"[red]Fund {args.fund!r} not found. Available: {', '.join(s.fund.name for s in result.snapshots)}"
            )
            return 1
        activity = fund_activity(snap).head(args.top)
        out = Table(
            title=f"{snap.fund.name} ({snap.fund.manager}) - 13F period {snap.report_date}, filed {snap.filing_date}"
        )
        for col in ("Ticker", "Issuer", "Value", "Weight", "Change", "Shares chg."):
            out.add_column(col)
        for _, row in activity.iterrows():
            style = {"NEW": "bold green", "ADD": "green", "REDUCE": "yellow", "SOLD": "red"}.get(
                row["change"]
            )
            out.add_row(
                row["ticker"] or "-",
                escape(str(row["issuer"])[:30]),
                _usd(row["value"]),
                _pct(row["weight"]),
                _styled(row["change"], style),
                _pct(row["share_change"], True),
            )
        console.print(out)
        return 0

    def render(title: str, frame: pd.DataFrame) -> None:
        out = Table(title=title)
        for col in (
            "Ticker",
            "Issuer",
            "Funds",
            "New",
            "Add",
            "Reduce",
            "Sold",
            "Value",
            "Avg weight",
        ):
            out.add_column(col)
        for _, row in frame.iterrows():
            out.add_row(
                row["ticker"] or "-",
                escape(str(row["issuer"])[:28]),
                str(row["holders"]),
                str(row["new"]),
                str(row["added"]),
                str(row["reduced"]),
                str(row["sold"]),
                _usd(row["total_value"]),
                _pct(row["avg_weight"]),
            )
        console.print(out)

    funds = ", ".join(f"{s.fund.name} ({s.report_date})" for s in result.snapshots)
    console.print(f"[bold]Funds analysed:[/] {funds}")
    render("Most owned by the tracked funds", result.consensus.head(args.top))
    render(
        "Most bought last quarter (new + added - reduced - sold)",
        most_bought(result.consensus, args.top),
    )
    render("Most sold last quarter", most_sold(result.consensus, min(args.top, 10)))
    for warning in result.warnings:
        console.print(_styled(f"! {warning}", "yellow"))
    console.print(
        "[dim]13F filings are published up to 45 days after quarter end and show long positions only.[/dim]"
    )
    return 0


def _single_stock(ticker: str, args: argparse.Namespace, config: dict):
    from screener.screener import run_screen

    provider = get_provider("demo" if args.demo else "yahoo", config)
    universe = "demo" if args.demo else args.universe
    with console.status(f"Screening {universe} to rank {ticker}..."):
        result = run_screen(universe, provider, config, args.profile)
    if ticker not in result.table.index:
        console.print(f"[red]{ticker} is not in universe {universe!r} (or has too little data).")
        return None, None
    return result, provider


def cmd_plan(args: argparse.Namespace, config: dict) -> int:
    ticker = args.ticker.upper()
    result, _ = _single_stock(ticker, args, config)
    if result is None:
        return 1
    row, plan = result.table.loc[ticker], result.plans[ticker]
    console.rule(escape(f"{ticker} - {row['name']}"))
    console.print(
        f"Action: {_styled(plan.action, ACTION_STYLE.get(plan.action))}   "
        f"Opportunity {plan.opportunity_score:.0f}/100   Technical {plan.technical_score:+.0f}   "
        f"Risk {_styled(row['risk'], RISK_STYLE.get(row['risk']))} ({row['risk_points']} pts)"
    )
    console.print(
        f"Price {_num(row['price'])}   Analyst target {_num(plan.analyst_target)} "
        f"({_pct(row['analyst_upside'], True)})   Dividend yield {_pct(row['dividend_yield'])}"
    )
    console.print("\n[bold]Why[/]")
    for reason in plan.reasons:
        console.print(f"  {escape(reason)}")
    console.print("\n[bold]Plan[/]")
    for rule in plan.exit_rules:
        console.print(f"  - {escape(rule)}")
    console.print("\n[bold]Factor scores[/] (percentile in universe)")
    for factor in ("value", "quality", "growth", "momentum", "income", "upside", "smart_money"):
        value = row.get(f"{factor}_score", math.nan)
        console.print(f"  {factor:<12} {'-' if math.isnan(value) else f'{value:5.0f}'}")
    console.print(DISCLAIMER)
    return 0


def cmd_backtest(args: argparse.Namespace, config: dict) -> int:
    from screener.analysis.backtest import run_backtest

    ticker = args.ticker.upper()
    provider = get_provider("demo" if args.demo else "yahoo", config)
    # one extra year so the indicators (SMA200, 12-month momentum) are warm at the start
    with console.status(f"Downloading {args.years + 1}y of prices for {ticker}..."):
        history = provider.get_history([ticker], f"{args.years + 1}y").get(ticker)
    if history is None:
        console.print(f"[red]No price data for {ticker}")
        return 1
    bt_cfg = dict(config["backtest"], cost_bps=args.cost_bps)
    result = run_backtest(history, config["signals"], bt_cfg)

    out = Table(
        title=f"Backtest {ticker}: {result.equity.index[0].date()} -> {result.equity.index[-1].date()}"
    )
    out.add_column("Metric")
    out.add_column("Strategy", justify="right")
    out.add_column("Buy & Hold", justify="right")
    fmt = {
        "total_return": _pct,
        "cagr": _pct,
        "volatility": _pct,
        "max_drawdown": _pct,
        "exposure": _pct,
        "win_rate": _pct,
        "avg_trade": _pct,
        "trades": lambda v: f"{v:.0f}",
    }
    for metric, values in result.stats.iterrows():
        f = fmt.get(metric, lambda v: _num(v, 2))
        out.add_row(metric, f(values["Strategy"]), f(values["Buy & Hold"]))
    console.print(out)
    if not result.trades.empty:
        console.print(result.trades.tail(args.last_trades).to_string(index=False))
    console.print(
        "[dim]Signals on close, fills at next open, costs included. Past performance does not "
        "predict future results.[/dim]"
    )
    return 0


def cmd_universes(_args: argparse.Namespace, config: dict) -> int:
    for name, description in available_universes(config).items():
        console.print(f"[bold]{name:<16}[/] {description}")
    console.print("You can also pass the path of a CSV file with a 'ticker' column.")
    return 0


# ------------------------------------------------------------------- parser
def build_parser(config: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="screener", description="Smart Money Stock Screener")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", help="YAML file overriding the default configuration")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, universe: bool = True) -> None:
        p.add_argument(
            "--demo", action="store_true", help="offline synthetic data, no internet needed"
        )
        if universe:
            p.add_argument(
                "-u", "--universe", default="us_large_caps", help="universe name or CSV path"
            )
            p.add_argument("-p", "--profile", default="balanced", choices=list(config["profiles"]))

    scan = sub.add_parser("scan", help="rank a universe of stocks")
    common(scan)
    scan.add_argument("--risk", nargs="+", choices=["Low", "Medium", "High"])
    scan.add_argument("--action", nargs="+", help='e.g. --action "STRONG BUY" BUY')
    scan.add_argument("--min-score", type=float, default=0.0, help="minimum opportunity score")
    scan.add_argument("--top", type=int, default=20)
    scan.add_argument("--smart-money", action="store_true", help="add 13F holdings (slower)")
    scan.add_argument("--csv", help="save the full filtered table to CSV")
    scan.set_defaults(func=cmd_scan)

    sm = sub.add_parser("smart-money", help="what the big investors hold and buy (13F)")
    common(sm, universe=False)
    sm.add_argument("--top", type=int, default=20)
    sm.add_argument("--fund", help="show the portfolio and changes of one fund")
    sm.set_defaults(func=cmd_smart_money)

    plan = sub.add_parser("plan", help="trade plan for one stock (entry, stop, target)")
    plan.add_argument("ticker")
    common(plan)
    plan.set_defaults(func=cmd_plan)

    bt = sub.add_parser("backtest", help="backtest the timing rules on one stock")
    bt.add_argument("ticker")
    common(bt, universe=False)
    bt.add_argument("--years", type=int, default=config["backtest"]["years"])
    bt.add_argument("--cost-bps", type=float, default=config["backtest"]["cost_bps"])
    bt.add_argument("--last-trades", type=int, default=10)
    bt.set_defaults(func=cmd_backtest)

    uni = sub.add_parser("universes", help="list the available universes")
    uni.set_defaults(func=cmd_universes)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # read --config before building the parser (profiles come from the config)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    config = load_config(known.config)

    args = build_parser(config).parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    try:
        return args.func(args, config)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # friendly message instead of a traceback
        if args.verbose:
            raise
        console.print(f"[red]Error:[/] {escape(str(exc))}  (run with -v for details)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
