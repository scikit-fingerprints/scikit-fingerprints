from typing import Union

import numpy as np
from scipy.sparse import csr_array


def _check_nan(arr: Union[np.ndarray, csr_array]) -> None:
    if isinstance(arr, np.ndarray):
        if np.isnan(arr).any():
            raise ValueError("Input array contains NaN values")
    elif isinstance(arr, csr_array):
        if np.isnan(arr.data).any():
            raise ValueError("Input sparse matrix contains NaN values")
    else:
        raise TypeError(
            f"Expected numpy.ndarray or scipy.sparse.csr_array, got {type(arr)}"
        )
