"""DataCache round-trip tests."""

import pandas as pd

from ccquant.data.cache import DataCache


def test_df_round_trip(tmp_path) -> None:
    cache = DataCache(root=tmp_path)
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=pd.RangeIndex(3, name="i"))

    assert cache.get_df("prices/TEST") is None
    cache.put_df("prices/TEST", df)
    assert cache.has("prices/TEST")

    out = cache.get_df("prices/TEST")
    pd.testing.assert_frame_equal(out, df)


def test_obj_round_trip(tmp_path) -> None:
    cache = DataCache(root=tmp_path)
    obj = {"universe": ["AAPL", "MSFT"], "n": 2}
    cache.put_obj("meta/universe", obj)
    assert cache.get_obj("meta/universe") == obj


def test_clear(tmp_path) -> None:
    cache = DataCache(root=tmp_path)
    cache.put_df("a", pd.DataFrame({"x": [1]}))
    cache.put_obj("b", [1, 2, 3])
    removed = cache.clear()
    assert removed == 2
    assert not cache.has("a")
    assert not cache.has("b")
