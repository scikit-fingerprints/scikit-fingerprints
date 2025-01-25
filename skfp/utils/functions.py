from collections.abc import Sequence

import pandas as pd


def get_data_from_indices(data: Sequence, indices: Sequence[int]) -> list:
    """
    Retrieve elements from ``data`` at specified ``indices``. Works not only
    for Python lists but also for e.g. NumPy arrays and Pandas DataFrames and
    Series.
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return [data.iloc[idx] for idx in indices]
    else:
        return [data[idx] for idx in indices]
