"""Point-in-time historical universe provider + membership mask tests."""

import pandas as pd

from ccquant.data.cache import DataCache
from ccquant.data.universe import HistoricalSP500Provider, build_membership_mask


def _write_csv(tmp_path):
    # date -> snapshot of members on that date (membership changes over time)
    csv = tmp_path / "hist.csv"
    csv.write_text(
        "date,tickers\n"
        "2010-01-01,\"AAA,BBB,CCC\"\n"
        "2015-06-01,\"AAA,CCC,DDD\"\n"      # BBB leaves, DDD joins
        "2020-03-01,\"AAA,DDD,EEE,BRK.B\"\n"  # CCC leaves, EEE + BRK.B join
    )
    return csv


def _provider(tmp_path):
    # isolated cache so snapshots from other tests don't leak in
    return HistoricalSP500Provider(source=_write_csv(tmp_path), cache=DataCache(tmp_path / "c"))


def test_point_in_time_membership(tmp_path):
    p = _provider(tmp_path)
    assert p.get_constituents("2012-01-01") == ["AAA", "BBB", "CCC"]
    assert p.get_constituents("2016-01-01") == ["AAA", "CCC", "DDD"]
    # BRK.B normalised to BRK-B for Yahoo
    assert p.get_constituents("2021-01-01") == ["AAA", "DDD", "EEE", "BRK-B"]


def test_before_first_snapshot_is_empty(tmp_path):
    p = _provider(tmp_path)
    assert p.get_constituents("2005-01-01") == []


def test_all_members_union(tmp_path):
    p = _provider(tmp_path)
    members = p.all_members("2010-01-01", "2021-01-01")
    assert set(members) == {"AAA", "BBB", "CCC", "DDD", "EEE", "BRK-B"}


def test_build_membership_mask(tmp_path):
    p = _provider(tmp_path)
    dates = pd.to_datetime(["2012-01-31", "2016-01-31"])
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    mask = build_membership_mask(p, dates, tickers)
    # 2012: AAA,BBB,CCC in; DDD out
    assert mask.loc[dates[0]].tolist() == [True, True, True, False]
    # 2016: AAA,CCC,DDD in; BBB out
    assert mask.loc[dates[1]].tolist() == [True, False, True, True]
