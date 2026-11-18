"""Smart Money Screener - Streamlit dashboard.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    import screener  # noqa: F401
except ModuleNotFoundError:  # running from a fresh clone without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).parent / "src"))

from screener.analysis.backtest import run_backtest
from screener.analysis.risk import classify_risk
from screener.analysis.signals import ACTIONS, BUY_ACTIONS
from screener.config import load_config
from screener.data import get_provider
from screener.data.universe import available_universes
from screener.screener import filter_table, run_screen
from screener.smart_money import fund_activity, most_bought, most_sold, run_smart_money

st.set_page_config(page_title="Smart Money Screener", page_icon="📈", layout="wide")

CONFIG = load_config()
ROOT = Path(__file__).parent

# Palette: categorical slots for series, status colours only for risk/up-down,
# always paired with a text label.
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#6250d6"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
RISK_COLORS = {"Low": GOOD, "Medium": WARNING, "High": CRITICAL}
RISK_LABEL = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}
ACTION_LABEL = {
    "STRONG BUY": "🟢🟢 STRONG BUY",
    "BUY": "🟢 BUY",
    "BUY ON PULLBACK": "🟡 BUY ON PULLBACK",
    "WATCH": "🔵 WATCH",
    "HOLD": "⚪ HOLD",
    "SELL": "🔴 SELL",
    "STRONG SELL": "🔴🔴 STRONG SELL",
    "N/A": "⚫ N/A",
}
PROFILE_HELP = {
    "balanced": "Mix of value, quality, growth, momentum and income",
    "income": "Dividends first (rendita): yield, quality, low leverage",
    "growth": "Fast-growing companies with strong price momentum",
    "value": "Cheap stocks vs earnings, cash flow and book value",
    "smart_money": "What the tracked superinvestors hold and buy",
}
PCT = st.column_config.NumberColumn(format="percent")
# show missing values as a dash where the installed Streamlit supports it
EMPTY = {"placeholder": "–"} if "placeholder" in inspect.signature(st.dataframe).parameters else {}


# ------------------------------------------------------------------ caching
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_smart_money(demo: bool):
    return run_smart_money(CONFIG, demo=demo)


@st.cache_data(ttl=3600, show_spinner=False)
def load_screen(universe: str, demo: bool, profile: str, use_smart_money: bool):
    provider = get_provider("demo" if demo else "yahoo", CONFIG)
    holders = load_smart_money(demo).holders_by_ticker() if use_smart_money else None
    return run_screen("demo" if demo else universe, provider, CONFIG, profile, holders)


@st.cache_data(ttl=3600, show_spinner=False)
def load_backtest(ticker: str, demo: bool, years: int, bt_cfg: tuple):
    provider = get_provider("demo" if demo else "yahoo", CONFIG)
    history = provider.get_history([ticker], f"{years + 1}y").get(ticker)
    if history is None:
        raise ValueError(f"No price data for {ticker}")
    history = history.iloc[-(years + 1) * 252 :]
    return run_backtest(history, CONFIG["signals"], dict(bt_cfg))


def fmt(x, kind: str = "num") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if kind == "pct":
        return f"{x:+.1%}"
    if kind == "pct0":
        return f"{x:.1%}"
    return f"{x:,.2f}"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📈 Smart Money Screener")
    mode = st.radio(
        "Data source",
        ["Demo (offline, synthetic)", "Live (Yahoo Finance + SEC)"],
        help="Demo runs instantly without internet. Live downloads real market data.",
    )
    demo = mode.startswith("Demo")
    universes = {k: v for k, v in available_universes(CONFIG).items() if k != "demo"}
    universe = st.selectbox(
        "Universe", list(universes), format_func=lambda k: f"{k} - {universes[k]}", disabled=demo
    )
    profile = st.selectbox(
        "Investor profile",
        list(CONFIG["profiles"]),
        format_func=lambda p: p.replace("_", " ").title(),
        help="Weights of the opportunity score",
    )
    st.caption(PROFILE_HELP.get(profile, ""))
    use_sm = st.toggle(
        "Include superinvestor 13F data",
        value=demo or profile == "smart_money",
        help="Live: first download takes 1-3 minutes, then it is cached.",
    )
    if st.button("🔄 Refresh data", width="stretch"):
        st.cache_data.clear()
    st.divider()
    st.caption(
        "⚠️ Educational project, **not investment advice**. Signals are rule-based, data can be "
        "delayed or wrong, and past performance does not predict future results."
    )

# --------------------------------------------------------------------- data
try:
    with st.spinner(
        "Screening... (live mode: the first run downloads data and can take a few minutes)"
    ):
        result = load_screen(universe, demo, profile, use_sm)
except Exception as exc:  # network, rate limit, bad config...
    st.error(f"Could not run the screener: {exc}")
    st.info("Tip: switch to **Demo** mode in the sidebar to explore the app offline.")
    st.stop()

table = result.table
st.title("Smart Money Stock Screener")
st.caption(
    f"Universe **{result.universe.name}** ({len(table)} stocks) · profile **{profile}** · "
    f"source **{result.provider}** · updated {result.generated_at:%Y-%m-%d %H:%M}"
    + (" · ⚠️ demo data: fictional companies" if demo else "")
)

tab_screen, tab_sm, tab_stock, tab_bt, tab_method = st.tabs(
    ["🎯 Screener", "🐋 Smart Money", "🔍 Stock analysis", "🧪 Backtest", "📖 Methodology"]
)

# ----------------------------------------------------------------- screener
with tab_screen:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Stocks screened", len(table))
    k2.metric("Buy signals", int(table["action"].isin(BUY_ACTIONS).sum()))
    k3.metric("Sell signals", int(table["action"].isin(["SELL", "STRONG SELL"]).sum()))
    k4.metric("Low-risk stocks", int((table["risk"] == "Low").sum()))
    k5.metric("High-risk stocks", int((table["risk"] == "High").sum()))

    f1, f2, f3, f4 = st.columns([1, 1.4, 1.4, 1])
    risk_filter = f1.multiselect("Risk", ["Low", "Medium", "High"])
    action_filter = f2.multiselect("Action", [a for a in ACTIONS if a in set(table["action"])])
    sector_filter = f3.multiselect("Sector", sorted(table["sector"].dropna().unique()))
    min_score = f4.slider("Min. opportunity", 0, 100, 0, step=5)
    view = filter_table(table, risk_filter, action_filter, sector_filter, min_score)

    display = view.assign(
        action=view["action"].map(ACTION_LABEL), risk=view["risk"].map(RISK_LABEL)
    ).reset_index()
    columns = [
        "ticker",
        "name",
        "sector",
        "action",
        "opportunity_score",
        "technical_score",
        "risk",
        "price",
        "entry",
        "stop_loss",
        "take_profit",
        "analyst_upside",
        "dividend_yield",
        "ret_12m",
        "volatility",
        "max_drawdown",
        "smart_money_holders",
    ]
    st.dataframe(
        display[columns],
        hide_index=True,
        **EMPTY,
        height=min(38 * (len(display) + 1), 640),
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", pinned=True),
            "name": "Name",
            "sector": "Sector",
            "action": "Action",
            "opportunity_score": st.column_config.ProgressColumn(
                "Opportunity",
                min_value=0,
                max_value=100,
                format="%.0f",
                help="Percentile of the factor composite in this universe",
            ),
            "technical_score": st.column_config.NumberColumn(
                "Timing", format="%+.0f", help="-100 bearish ... +100 bullish"
            ),
            "risk": "Risk",
            "price": st.column_config.NumberColumn("Price", format="%.2f"),
            "entry": st.column_config.NumberColumn("Entry", format="%.2f"),
            "stop_loss": st.column_config.NumberColumn("Stop-loss", format="%.2f"),
            "take_profit": st.column_config.NumberColumn("Take profit", format="%.2f"),
            "analyst_upside": st.column_config.NumberColumn("Analyst upside", format="percent"),
            "dividend_yield": st.column_config.NumberColumn("Div. yield", format="percent"),
            "ret_12m": st.column_config.NumberColumn("12m return", format="percent"),
            "volatility": st.column_config.NumberColumn("Volatility", format="percent"),
            "max_drawdown": st.column_config.NumberColumn("Max DD 1y", format="percent"),
            "smart_money_holders": st.column_config.NumberColumn(
                "Funds", format="%.0f", help="Tracked superinvestors holding it"
            ),
        },
    )
    st.download_button(
        "⬇️ Download CSV",
        view.to_csv().encode(),
        f"screener_{result.universe.name}_{profile}.csv",
        "text/csv",
    )

    st.subheader("Risk vs opportunity map")
    st.caption("Top-left = high opportunity with low volatility. Hover a dot for details.")
    fig = go.Figure()
    for level in ("Low", "Medium", "High"):
        part = view[view["risk"] == level]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["volatility"],
                y=part["opportunity_score"],
                mode="markers",
                name=f"{level} risk",
                marker=dict(size=11, color=RISK_COLORS[level], line=dict(width=2, color="white")),
                customdata=np.stack([part.index, part["name"], part["action"]], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}<br>%{customdata[2]}<br>"
                "Volatility %{x:.1%}<br>Opportunity %{y:.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=430,
        xaxis_title="Annualised volatility (1y)",
        yaxis_title="Opportunity score",
        xaxis_tickformat=".0%",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig)

    if result.warnings:
        with st.expander(f"⚠️ {len(result.warnings)} data warnings"):
            st.write("\n".join(f"- {w}" for w in result.warnings))

# -------------------------------------------------------------- smart money
with tab_sm:
    try:
        with st.spinner("Loading 13F filings..."):
            sm = load_smart_money(demo)
    except Exception as exc:
        st.error(f"Could not load 13F data: {exc}")
        sm = None

    if sm is not None and not sm.consensus.empty:
        st.caption(
            "Source: SEC Form 13F-HR. Funds disclose their US long positions up to **45 days after quarter end**: "
            "this is what they held, not necessarily what they hold today."
        )
        funds_df = pd.DataFrame(
            [
                {
                    "Fund": s.fund.name,
                    "Manager": s.fund.manager,
                    "Period": s.report_date,
                    "Filed": s.filing_date,
                    "Positions": len(s.current),
                    "Portfolio value": s.current["value"].sum(),
                }
                for s in sm.snapshots
            ]
        )
        with st.expander(f"Funds tracked ({len(funds_df)})"):
            st.dataframe(
                funds_df,
                hide_index=True,
                column_config={"Portfolio value": st.column_config.NumberColumn(format="compact")},
            )
        sm_cols = {
            "ticker": "Ticker",
            "issuer": "Issuer",
            "holders": st.column_config.NumberColumn("Funds holding"),
            "new": "New",
            "added": "Added",
            "reduced": "Reduced",
            "sold": "Sold out",
            "net_buyers": st.column_config.NumberColumn("Net buyers", format="%+d"),
            "total_value": st.column_config.NumberColumn("Total value $", format="compact"),
            "avg_weight": st.column_config.NumberColumn("Avg weight", format="percent"),
            "max_weight": st.column_config.NumberColumn("Max weight", format="percent"),
            "funds": "Held by",
        }
        s1, s2, s3, s4 = st.tabs(["Most owned", "Most bought", "Most sold", "By fund"])
        with s1:
            top = sm.consensus.head(25)
            bar = top.head(15).iloc[::-1]
            labels = bar["ticker"].fillna(bar["issuer"].str.title())
            fig = go.Figure(
                go.Bar(
                    x=bar["holders"],
                    y=labels,
                    orientation="h",
                    marker_color=BLUE,
                    hovertemplate="%{y}: %{x} funds<extra></extra>",
                )
            )
            fig.update_layout(
                height=420,
                xaxis=dict(title="Number of tracked funds holding the stock", dtick=1),
                margin=dict(l=10, r=10, t=10, b=10),
                bargap=0.35,
            )
            st.plotly_chart(fig)
            st.dataframe(top, hide_index=True, column_config=sm_cols, **EMPTY)
        with s2:
            st.caption(
                "Ranked by net buyers = new positions + increases - reductions - full exits."
            )
            st.dataframe(
                most_bought(sm.consensus, 25), hide_index=True, column_config=sm_cols, **EMPTY
            )
        with s3:
            st.dataframe(
                most_sold(sm.consensus, 25), hide_index=True, column_config=sm_cols, **EMPTY
            )
        with s4:
            names = [s.fund.name for s in sm.snapshots]
            chosen = st.selectbox("Fund", names)
            snap = sm.snapshots[names.index(chosen)]
            st.caption(
                f"{snap.fund.manager} · period {snap.report_date} · filed {snap.filing_date}"
            )
            st.dataframe(
                fund_activity(snap),
                hide_index=True,
                **EMPTY,
                column_config={
                    "ticker": "Ticker",
                    "issuer": "Issuer",
                    "value": st.column_config.NumberColumn("Value $", format="compact"),
                    "shares": st.column_config.NumberColumn("Shares", format="compact"),
                    "weight": st.column_config.NumberColumn("Weight", format="percent"),
                    "prev_shares": st.column_config.NumberColumn("Prev. shares", format="compact"),
                    "share_change": st.column_config.NumberColumn("Share change", format="percent"),
                    "change": "Change",
                },
            )
        if sm.warnings:
            with st.expander(f"⚠️ {len(sm.warnings)} warnings"):
                st.write("\n".join(f"- {w}" for w in sm.warnings))
    elif sm is not None:
        st.info("No 13F data available.")


# ------------------------------------------------------------ stock analysis
def price_chart(ind: pd.DataFrame, plan, days: int) -> go.Figure:
    d = ind.iloc[-days:]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.04
    )
    fig.add_trace(
        go.Candlestick(
            x=d.index,
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            name="Price",
            increasing_line_color=GOOD,
            decreasing_line_color=CRITICAL,
        ),
        1,
        1,
    )
    fig.add_trace(
        go.Scatter(x=d.index, y=d["sma50"], name="SMA 50", line=dict(color=BLUE, width=2)), 1, 1
    )
    fig.add_trace(
        go.Scatter(x=d.index, y=d["sma200"], name="SMA 200", line=dict(color=ORANGE, width=2)), 1, 1
    )
    for level, label, color in (
        (plan.entry, "Entry", BLUE),
        (plan.stop_loss, "Stop-loss", CRITICAL),
        (plan.take_profit, "Take profit", GOOD),
    ):
        if level:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color=color,
                line_width=1.5,
                row=1,
                col=1,
                annotation_text=f"{label} {level:,.2f}",
                annotation_position="top left",
            )
    fig.add_trace(
        go.Scatter(x=d.index, y=d["rsi14"], name="RSI", line=dict(color=VIOLET, width=2)), 2, 1
    )
    for y in (30, 70):
        fig.add_hline(y=y, line_dash="dot", line_color="gray", line_width=1, row=2, col=1)
    hist_colors = np.where(d["macd_hist"] >= 0, GOOD, CRITICAL)
    fig.add_trace(
        go.Bar(x=d.index, y=d["macd_hist"], marker_color=hist_colors, name="MACD histogram"), 3, 1
    )
    fig.add_trace(
        go.Scatter(x=d.index, y=d["macd"], name="MACD", line=dict(color=BLUE, width=1.5)), 3, 1
    )
    fig.add_trace(
        go.Scatter(
            x=d.index, y=d["macd_signal"], name="Signal", line=dict(color=ORANGE, width=1.5)
        ),
        3,
        1,
    )
    fig.update_layout(
        height=720,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.07),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI 14", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


with tab_stock:
    tickers = list(table.index)
    ticker = st.selectbox(
        "Stock", tickers, format_func=lambda t: f"{t} - {table.loc[t, 'name']}", key="stock"
    )
    row, plan = table.loc[ticker], result.plans[ticker]

    st.subheader(f"{row['name']} ({ticker})  ·  {ACTION_LABEL[plan.action]}")
    m = st.columns(6)
    m[0].metric(
        "Price",
        fmt(row["price"]),
        None if math.isnan(row["ret_1m"]) else f"{row['ret_1m']:+.1%} 1m",
    )
    m[1].metric("Opportunity", f"{plan.opportunity_score:.0f}/100")
    m[2].metric("Timing score", f"{plan.technical_score:+.0f}")
    m[3].metric(
        "Risk", row["risk"], help=f"{row['risk_points']} risk points, see the breakdown below"
    )
    m[4].metric("Analyst upside", fmt(row["analyst_upside"], "pct"))
    m[5].metric("Dividend yield", fmt(row["dividend_yield"], "pct0"))

    period = st.segmented_control("Chart period", ["3M", "6M", "1Y", "2Y"], default="1Y")
    days = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}[period or "1Y"]
    st.plotly_chart(price_chart(result.indicators[ticker], plan, days))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Why this signal")
        for reason in plan.reasons:
            icon = {"+": "✅", "-": "❌", "=": "➖"}.get(reason[0], "•")
            st.markdown(f"{icon} {reason[2:]}")
    with c2:
        st.markdown("#### When to buy / when to sell")
        for rule in plan.exit_rules:
            st.markdown(f"- {rule}")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Factor scores")
        factors = ["value", "quality", "growth", "momentum", "income", "upside", "smart_money"][
            ::-1
        ]
        scores = [row.get(f"{f}_score", math.nan) for f in factors]
        fig = go.Figure(
            go.Bar(
                x=scores,
                y=[f.replace("_", " ").title() for f in factors],
                orientation="h",
                marker_color=BLUE,
                hovertemplate="%{y}: %{x:.0f}<extra></extra>",
            )
        )
        fig.update_layout(
            height=300,
            xaxis=dict(range=[0, 100], title="Percentile in universe"),
            margin=dict(l=10, r=10, t=10, b=10),
            bargap=0.35,
        )
        st.plotly_chart(fig)
    with c4:
        st.markdown("#### Risk breakdown")
        risk = classify_risk(
            row["volatility"],
            row["beta"],
            row["max_drawdown"],
            row["debt_to_equity"],
            row["market_cap"],
            CONFIG["risk"],
        )
        values = {
            "volatility": fmt(row["volatility"], "pct0"),
            "beta": fmt(row["beta"]),
            "max_drawdown": fmt(row["max_drawdown"], "pct"),
            "debt_to_equity": fmt(row["debt_to_equity"]),
            "market_cap": f"{row['market_cap'] / 1e9:,.1f}B"
            if not math.isnan(row["market_cap"])
            else "-",
        }
        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": [k.replace("_", " ").capitalize() for k in risk.breakdown],
                    "Value": [values[k] for k in risk.breakdown],
                    "Risk points": list(risk.breakdown.values()),
                }
            ),
            hide_index=True,
        )
        st.caption(
            f"Total {risk.points}/{risk.max_points} points → **{risk.level}**. "
            "Thresholds are in config.yaml (volatility counts double)."
        )

# ------------------------------------------------------------------ backtest
with tab_bt:
    st.caption(
        "Tests the **timing rules** only (prices). Signal on the close, order filled at the next open, "
        "transaction costs included. Fundamentals are not backtested: free sources only provide today's "
        "values, using them in the past would be look-ahead bias."
    )
    b1, b2, b3 = st.columns([2, 1, 1])
    bt_ticker = b1.selectbox(
        "Stock", tickers, format_func=lambda t: f"{t} - {table.loc[t, 'name']}", key="bt"
    )
    years = b2.slider("Years", 2, 4 if demo else 10, 3 if demo else 5)
    cost = b3.number_input(
        "Cost per trade (bps)", 0.0, 100.0, float(CONFIG["backtest"]["cost_bps"]), step=5.0
    )
    with st.expander("Strategy parameters"):
        p1, p2, p3 = st.columns(3)
        entry_th = p1.slider(
            "Enter when timing score ≥", -20, 100, CONFIG["backtest"]["entry_threshold"], step=20
        )
        exit_th = p2.slider(
            "Exit when timing score ≤", -100, 20, CONFIG["backtest"]["exit_threshold"], step=20
        )
        trail = p3.slider(
            "Trailing stop (× ATR)",
            1.0,
            6.0,
            float(CONFIG["backtest"]["trailing_atr_multiplier"]),
            0.5,
        )

    bt_cfg = (
        ("entry_threshold", entry_th),
        ("exit_threshold", exit_th),
        ("trailing_atr_multiplier", trail),
        ("cost_bps", cost),
    )
    try:
        bt = load_backtest(bt_ticker, demo, years, bt_cfg)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        bt = None

    if bt is not None:
        s = bt.stats
        st.markdown("**Strategy** (difference vs buy & hold below each value)")
        strat, bh = s["Strategy"], s["Buy & Hold"]
        k = st.columns(4)
        k[0].metric(
            "CAGR",
            fmt(strat["cagr"], "pct"),
            f"{(strat['cagr'] - bh['cagr']) * 100:+.1f} pp",
            help=f"Compound annual growth. Buy & hold: {fmt(bh['cagr'], 'pct')}",
        )
        k[1].metric(
            "Max drawdown",
            fmt(strat["max_drawdown"], "pct"),
            f"{(strat['max_drawdown'] - bh['max_drawdown']) * 100:+.1f} pp",
            help=f"Worst peak-to-trough loss. Buy & hold: {fmt(bh['max_drawdown'], 'pct')}",
        )
        k[2].metric(
            "Sharpe ratio",
            fmt(strat["sharpe"]),
            f"{strat['sharpe'] - bh['sharpe']:+.2f}",
            help=f"Return per unit of volatility. Buy & hold: {fmt(bh['sharpe'])}",
        )
        k[3].metric("Trades", f"{strat['trades']:.0f}", help="Number of round trips")
        st.caption(
            f"Time invested: {strat['exposure']:.0%} of the days · "
            f"win rate {fmt(strat['win_rate'], 'pct0')} · average trade {fmt(strat['avg_trade'], 'pct')}"
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bt.equity.index, y=bt.equity, name="Strategy", line=dict(color=BLUE, width=2)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bt.buy_hold.index,
                y=bt.buy_hold,
                name="Buy & Hold",
                line=dict(color=ORANGE, width=2),
            )
        )
        fig.update_layout(
            height=380,
            yaxis_title="Growth of 1",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig)

        if not bt.trades.empty:
            trades = bt.trades
            price = bt.buy_hold  # close / first close
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=price.index,
                    y=price,
                    name="Price (indexed)",
                    line=dict(color="gray", width=1.5),
                )
            )
            entries = pd.to_datetime(trades["entry_date"])
            exits = pd.to_datetime(trades.loc[trades["status"] == "closed", "exit_date"])
            fig.add_trace(
                go.Scatter(
                    x=entries,
                    y=price.reindex(entries, method="nearest"),
                    mode="markers",
                    name="Buy",
                    marker=dict(
                        symbol="triangle-up", size=11, color=GOOD, line=dict(width=1, color="white")
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=exits,
                    y=price.reindex(exits, method="nearest"),
                    mode="markers",
                    name="Sell",
                    marker=dict(
                        symbol="triangle-down",
                        size=11,
                        color=CRITICAL,
                        line=dict(width=1, color="white"),
                    ),
                )
            )
            fig.update_layout(
                height=320,
                yaxis_title="Price (indexed)",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig)

        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.markdown("#### Statistics")
            st.dataframe(
                s,
                column_config={
                    "Strategy": st.column_config.NumberColumn(format="%.3f"),
                    "Buy & Hold": st.column_config.NumberColumn(format="%.3f"),
                },
            )
        with c2:
            st.markdown("#### Trades")
            st.dataframe(
                bt.trades.iloc[::-1],
                hide_index=True,
                column_config={
                    "return": st.column_config.NumberColumn("Net return", format="percent")
                },
            )

# --------------------------------------------------------------- methodology
with tab_method:
    doc = ROOT / "docs" / "METHODOLOGY.md"
    st.markdown(doc.read_text(encoding="utf-8") if doc.exists() else "See docs/METHODOLOGY.md")
    if result.weights:
        st.markdown("#### Weights used by the current profile")
        st.dataframe(
            pd.DataFrame({"Factor": list(result.weights), "Weight": list(result.weights.values())}),
            hide_index=True,
            column_config={"Weight": PCT},
        )
