"""Tiny disk cache with time-to-live.

Free data sources (Yahoo Finance, SEC EDGAR, OpenFIGI) are rate limited, so
everything that is downloaded is cached locally. Values are pickled: the
cache is a private, local directory written only by this program.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any


class DiskCache:
    def __init__(self, directory: str | Path, default_ttl_hours: float = 12.0) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl_hours * 3600

    def _path(self, key: str) -> Path:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
        return self.directory / f"{digest}.pkl"

    def get(self, key: str, ttl_hours: float | None = None) -> Any | None:
        """Return the cached value, or ``None`` if missing or expired.

        ``ttl_hours=float('inf')`` never expires (immutable data such as a
        filed 13F report).
        """
        path = self._path(key)
        if not path.exists():
            return None
        ttl = self.default_ttl if ttl_hours is None else ttl_hours * 3600
        if time.time() - path.stat().st_mtime > ttl:
            return None
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except (OSError, pickle.UnpicklingError, EOFError):
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def clear(self) -> int:
        removed = 0
        for file in self.directory.glob("*.pkl"):
            file.unlink(missing_ok=True)
            removed += 1
        return removed
