import numpy as np
import pandas as pd

from skfp.utils import get_data_from_indices


def test_get_data_from_indices_numpy():
    data = np.array([1, 2, 3, 4])
    idxs = [0, 2]
    selected_data = get_data_from_indices(data, idxs)
    assert selected_data == [1, 3]


def test_get_data_from_indices_pandas():
    data = pd.Series([1, 2, 3, 4])
    idxs = [0, 2]
    selected_data = get_data_from_indices(data, idxs)
    assert selected_data == [1, 3]


def test_get_data_from_indices_pandas_str_index():
    data = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
    idxs = [0, 2]
    selected_data = get_data_from_indices(data, idxs)
    assert selected_data == [1, 3]
