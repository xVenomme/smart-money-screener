"""Smart Money Screener.

Screens stocks by combining:
* what the big institutional investors hold and buy (SEC Form 13F),
* a factor-based opportunity score (value, quality, growth, momentum, income...),
* a transparent risk classification (Low / Medium / High),
* rule-based timing signals with entry, stop-loss and take-profit levels,
* a walk-forward backtest of those timing rules.

Educational project - not investment advice.
"""

__version__ = "0.1.0"
