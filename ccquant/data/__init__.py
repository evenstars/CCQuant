"""Data layer: loading, adjustment, universe construction, caching.

M0 ships the cache framework only. Loaders (yfinance / Polygon / Norgate) and
universe construction arrive in M1.
"""

from ccquant.data.cache import DataCache

__all__ = ["DataCache"]
