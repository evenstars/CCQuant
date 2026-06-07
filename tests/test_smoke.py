"""Smoke tests: the empty pipeline imports and wires together (M0 deliverable)."""

import importlib

import pytest

MODULES = [
    "ccquant",
    "ccquant.utils",
    "ccquant.utils.paths",
    "ccquant.utils.config",
    "ccquant.utils.logging",
    "ccquant.data",
    "ccquant.data.cache",
    "ccquant.strategy",
    "ccquant.backtest",
    "ccquant.risk",
    "ccquant.live",
    "ccquant.monitor",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_version() -> None:
    import ccquant

    assert ccquant.__version__


def test_logger_is_idempotent() -> None:
    from ccquant.utils.logging import get_logger

    log = get_logger("ccquant.test")
    n_handlers = len(log.handlers)
    log2 = get_logger("ccquant.test")
    assert log is log2
    assert len(log2.handlers) == n_handlers  # no duplicate handlers
