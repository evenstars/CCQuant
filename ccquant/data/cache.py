"""Local on-disk cache for market data.

A thin key/value store over the filesystem so repeated runs don't re-hit data
providers. Frames are stored as Parquet (fast, typed); arbitrary objects fall
back to pickle. Keys are namespaced and hashed into safe filenames.

This is the M0 skeleton: the storage mechanics are real and tested, while the
data *loaders* that populate it land in M1.

Example:
    cache = DataCache()
    if (df := cache.get_df("prices/AAPL")) is None:
        df = download_prices("AAPL")          # M1
        cache.put_df("prices/AAPL", df)
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from ccquant.utils import paths
from ccquant.utils.logging import get_logger

_log = get_logger("ccquant.data.cache")


class DataCache:
    """Filesystem-backed cache for DataFrames and pickleable objects."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or paths.DATA_CACHE_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    # -- key handling ------------------------------------------------------- #
    def _path(self, key: str, suffix: str) -> Path:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        safe = key.replace("/", "__").replace(" ", "_")[:60]
        return self.root / f"{safe}.{digest}{suffix}"

    # -- DataFrame API ------------------------------------------------------ #
    def put_df(self, key: str, df: pd.DataFrame) -> Path:
        """Store a DataFrame as Parquet, falling back to pickle if no Parquet engine."""
        try:
            path = self._path(key, ".parquet")
            df.to_parquet(path)
        except (ImportError, ValueError):
            path = self._path(key, ".dfpkl")
            df.to_pickle(path)
            _log.debug("parquet unavailable; cached df as pickle -> %s", path.name)
        return path

    def get_df(self, key: str) -> pd.DataFrame | None:
        pq = self._path(key, ".parquet")
        if pq.exists():
            try:
                return pd.read_parquet(pq)
            except ImportError:
                pass  # fall through to a pickle copy if one exists
        pk = self._path(key, ".dfpkl")
        if pk.exists():
            return pd.read_pickle(pk)
        return None

    # -- generic object API ------------------------------------------------- #
    def put_obj(self, key: str, obj: Any) -> Path:
        path = self._path(key, ".pkl")
        with path.open("wb") as fh:
            pickle.dump(obj, fh)
        return path

    def get_obj(self, key: str) -> Any | None:
        path = self._path(key, ".pkl")
        if not path.exists():
            return None
        with path.open("rb") as fh:
            return pickle.load(fh)

    # -- maintenance -------------------------------------------------------- #
    def has(self, key: str) -> bool:
        return any(
            self._path(key, ext).exists() for ext in (".parquet", ".dfpkl", ".pkl")
        )

    def clear(self) -> int:
        """Delete all cache files. Returns the number of files removed."""
        n = 0
        for f in self.root.glob("*"):
            if f.is_file():
                f.unlink()
                n += 1
        _log.info("cleared %d cache files from %s", n, self.root)
        return n
