import numpy as np
import pytest

from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm


@td.skip_array_manager_invalid_test  # with ArrayManager df.loc[0] is not a view
def test_cache_updating(using_copy_on_write):
    # 5216
    # make sure that we don't try to set a dead cache
    a = np.random.rand(10, 3)
    df = DataFrame(a, columns=["x", "y", "z"])
    df_original = df.copy()
    tuples = [(i, j) for i in range(5) for j in range(2)]
    index = MultiIndex.from_tuples(tuples)
    df.index = index

    # setting via chained assignment
    # but actually works, since everything is a view
    df.loc[0]["z"].iloc[0] = 1.0
    result = df.loc[(0, 0), "z"]
    if using_copy_on_write:
        assert result == df_original.loc[0, "z"]
    else:
        assert result == 1

    # correct setting
    df.loc[(0, 0), "z"] = 2
    result = df.loc[(0, 0), "z"]
    assert result == 2


@pytest.mark.slow
def test_indexer_caching():
    # GH5727
    # make sure that indexers are in the _internal_names_set
    n = 1000001
    arrays = (range(n), range(n))
    index = MultiIndex.from_tuples(zip(*arrays))
    s = Series(np.zeros(n), index=index)
    str(s)

    # setitem
    expected = Series(np.ones(n), index=index)
    s = Series(np.zeros(n), index=index)
    s[s == 0] = 1
    tm.assert_series_equal(s, expected)
