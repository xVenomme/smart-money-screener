# Methodology

Every number in the screener comes from a rule written in plain code and
listed below. Nothing is a black box, and every threshold can be changed in
`config.yaml`.

## 1. Data

| What | Source | Notes |
|---|---|---|
| Daily prices (adjusted) | Yahoo Finance via `yfinance` | 2 years for the screener, up to 10 for the backtest |
| Fundamentals snapshot | Yahoo Finance via `yfinance` | P/E, EV/EBITDA, P/B, FCF, ROE, margins, debt/equity, growth, dividends, analyst targets |
| Institutional holdings | SEC EDGAR, Form 13F-HR | latest two quarters per fund |
| CUSIP → ticker | OpenFIGI API | free, rate limited, cached forever |

Everything downloaded is cached on disk (`.cache/`), so repeated runs are fast
and gentle on the free APIs.

## 2. Smart money: what the big investors buy

Managers with more than $100M in US equities must file a **Form 13F-HR**
every quarter, within 45 days of quarter end. The screener follows a list of
concentrated, long-term "superinvestors" (`funds.yaml`) and, for each one,
compares the latest two filings:

| Label | Rule (share count, 2% tolerance) |
|---|---|
| **NEW** | not held last quarter, held now |
| **ADD** | shares up more than 2% |
| **HOLD** | shares within ±2% |
| **REDUCE** | shares down more than 2% |
| **SOLD** | held last quarter, zero now |

The consensus table then counts, for every stock, how many funds hold it,
the total value, the average portfolio weight and the **net buyers**
(`new + added − reduced − sold`).

Data-quality details handled in code:

* the same CUSIP on several rows (sub-managers) is summed;
* options (`putCall`) and bonds (`PRN`) are excluded;
* some filers still report values **in thousands** even though the SEC
  switched to dollars in 2023: the unit is detected from the implied price
  per share (value / shares) instead of trusting the filing date;
* funds whose last filing is older than 200 days are skipped and reported.

## 3. Opportunity score (0–100): *what* to buy

1. Each metric becomes a **percentile inside the screened universe**
   (0 = worst, 100 = best). Missing values count as neutral (50).
2. Metrics are averaged into seven factors:

| Factor | Metrics | Better when |
|---|---|---|
| Value | forward P/E, EV/EBITDA, P/B, FCF yield | lower multiples, higher FCF yield; negative P/E (losses) = worst |
| Quality | ROE, profit margin, debt/equity | high profitability, low leverage |
| Growth | revenue growth, earnings growth | higher |
| Momentum | 12-1 month return, 6 month return | higher (12-1 skips the last month, which tends to mean-revert) |
| Income | dividend yield | higher; **halved if payout ratio > 100%** (yield trap) |
| Upside | analysts' mean target / price − 1 | higher |
| Smart money | funds holding, net buyers | higher |

3. Factors are combined with the weights of the selected **profile**
   (`balanced`, `income`, `growth`, `value`, `smart_money`). Factors without
   any data are dropped and the weights re-normalised.
4. The composite is ranked again, so the final score reads as *"better than
   X% of the stocks screened"*. It is a **relative** score: the best stock of a
   bad universe still scores 100.

## 4. Risk class: Low / Medium / High

A points system with **fixed thresholds** (not relative), so the label keeps
its meaning across markets:

| Metric (1 year) | 0 points | 1 point | 2 points |
|---|---|---|---|
| Annualised volatility (×2 weight) | ≤ 22% | ≤ 35% | > 35% |
| Beta vs benchmark | ≤ 0.9 | ≤ 1.3 | > 1.3 |
| Max drawdown | ≤ 20% | ≤ 35% | > 35% |
| Debt / equity | ≤ 100% | ≤ 200% | > 200% |
| Market cap | ≥ $50B | ≥ $10B | < $10B |

Total 0–12 points: **≤ 3 Low**, **≤ 7 Medium**, otherwise **High**. Volatility
above 45% is always High. Missing data scores 1 point: unknown is not safe.

## 5. Timing: *when* to buy and sell

### Technical score (−100 … +100)

Five votes of −1 / 0 / +1, averaged and multiplied by 100:

| Component | +1 | −1 |
|---|---|---|
| Trend | close > SMA200 | close < SMA200 |
| Regime | SMA50 > SMA200 (golden cross) | SMA50 < SMA200 (death cross) |
| MACD | MACD > signal line | MACD < signal line |
| RSI(14) | < 30 (oversold) | > 70 (overbought), 0 otherwise |
| Momentum | 12-1 month return > 0 | < 0 |

### Action

| Action | Rule (checked in this order) |
|---|---|
| STRONG SELL | technical ≤ −60 and opportunity < 40 |
| SELL | technical ≤ −20 |
| STRONG BUY | technical ≥ 60, opportunity ≥ 75, RSI ≤ 70 |
| BUY / BUY ON PULLBACK | technical ≥ 20 and opportunity ≥ 50 (pullback if RSI > 70) |
| WATCH | opportunity ≥ 60: good company, wait for the trend |
| HOLD | everything else |

### Trade plan for buy signals

* **Entry**: current price; for *BUY ON PULLBACK* a limit price at the
  20-day average or one ATR below the price.
* **Stop-loss**: entry − k × ATR(14), with k = 2 / 2.5 / 3 for Low / Medium /
  High risk. Volatile stocks get more room, so normal noise doesn't stop you out.
* **Take profit**: entry + 2 × (entry − stop), i.e. reward/risk 2:1.
* **Exit rules**: trailing stop at the highest close since entry − 3 × ATR;
  technical score ≤ −20; close below the 200-day average.

## 6. Backtest

The timing rules are tested on each stock against buy & hold:

* signals are computed on the **close** of day *t* using data up to *t* only;
* orders are filled at the **open of day t+1**;
* entry when technical score ≥ 20, exit when ≤ −20 or when the close drops
  below the trailing stop (highest close − 3 × ATR);
* 10 bps of costs per side, long only;
* one extra year of data is downloaded so the indicators are warm on day one.

A unit test checks that the decisions up to day *t* do not change when
future data is appended (**no look-ahead bias**).

Fundamentals are **not** backtested: free sources only provide today's
values, and using them in the past would be look-ahead bias. A
point-in-time fundamentals database would be needed for that.

## 7. Known limitations

* 13F data is delayed up to 45 days and shows long US positions only (no
  shorts, cash, bonds or foreign shares). Stock splits look like ADD.
* `yfinance` is an unofficial API: fields can be missing or change.
* The S&P 500 universe today contains the survivors: historical tests on it
  carry **survivorship bias**.
* Rule-based signals describe the current trend; they do not predict the
  future. Trend following usually lowers drawdowns in prolonged bear markets
  and lags buy & hold in strong bull markets - the backtest shows both.

**This is an educational project, not investment advice.**
