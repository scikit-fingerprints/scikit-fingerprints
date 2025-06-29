from collections.abc import Sequence
from importlib.metadata import version

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


def _get_sklearn_version():
    sklearn_ver = version("scikit-learn")  # e.g. 1.6.0
    sklearn_ver = ".".join(sklearn_ver.split(".")[:-1])  # e.g. 1.6
    return float(sklearn_ver)
