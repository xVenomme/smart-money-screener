"""Data access layer: market data providers, universes and SEC 13F filings."""

from screener.data.base import FUNDAMENTAL_FIELDS, MarketDataProvider

__all__ = ["FUNDAMENTAL_FIELDS", "MarketDataProvider", "get_provider"]


def get_provider(name: str, config: dict):
    """Factory: ``"yahoo"`` for live data, ``"demo"`` for synthetic offline data."""
    if name == "demo":
        from screener.data.demo import DemoProvider

        return DemoProvider()
    if name == "yahoo":
        from screener.cache import DiskCache
        from screener.data.yahoo import YahooProvider

        cache = DiskCache(config["data"]["cache_dir"], config["data"]["cache_ttl_hours"])
        return YahooProvider(cache=cache, max_workers=config["data"]["max_workers"])
    raise ValueError(f"Unknown data provider: {name!r} (use 'yahoo' or 'demo')")
