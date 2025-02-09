from typing import Union

import numpy as np
from scipy.sparse import csr_array


def _check_finite_values(arr: Union[np.ndarray, csr_array]) -> None:
    if isinstance(arr, np.ndarray):
        if not np.isfinite(arr).all():
            raise ValueError("Input array contains infinity or NaN values")
    elif isinstance(arr, csr_array):
        if not np.isfinite(arr.data).all():
            raise ValueError("Input sparse matrix contains infinity or NaN values")
    else:
        raise TypeError(
            f"Expected numpy.ndarray or scipy.sparse.csr_array, got {type(arr)}"
        )
