"""Universe provider tests (no network)."""

import pandas as pd

from ccquant.data.universe import StaticUniverseProvider, UniverseProvider


def test_static_provider_is_date_independent():
    p = StaticUniverseProvider(["AAA", "BBB", "AAA"])  # dupe dropped
    assert isinstance(p, UniverseProvider)
    assert p.get_constituents(pd.Timestamp("2005-01-01")) == ["AAA", "BBB"]
    assert p.get_constituents(pd.Timestamp("2026-01-01")) == ["AAA", "BBB"]


def test_static_provider_returns_copy():
    p = StaticUniverseProvider(["AAA"])
    got = p.get_constituents(pd.Timestamp("2020-01-01"))
    got.append("ZZZ")
    # mutating the returned list must not corrupt the provider
    assert p.get_constituents(pd.Timestamp("2020-01-01")) == ["AAA"]
