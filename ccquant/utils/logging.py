"""Project-wide logging setup.

Provides a single `get_logger` helper so every module logs in a consistent
format. Console handler always on; an optional rotating file handler writes to
`logs/ccquant.log` when `to_file=True`.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from ccquant.utils import paths

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_configured: set[str] = set()


def get_logger(
    name: str = "ccquant",
    level: int = logging.INFO,
    to_file: bool = False,
) -> logging.Logger:
    """Return a configured logger. Idempotent per logger name."""
    logger = logging.getLogger(name)
    if name in _configured:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if to_file:
        paths.LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            paths.LOG_DIR / "ccquant.log",
            maxBytes=5_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    _configured.add(name)
    return logger
