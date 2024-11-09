from collections.abc import Sequence

import pandas as pd


def get_data_from_indices(data: Sequence, indices: Sequence[int]) -> list:
    """
    Helper function to retrieve data elements from specified indices.
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return [data.iloc[idx] for idx in indices]
    else:
        return [data[idx] for idx in indices]
