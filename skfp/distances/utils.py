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


def _check_valid_vectors(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> None:
    if (
        type(vec_a) is not type(vec_b)
        or not isinstance(vec_a, (np.ndarray, csr_array))
        or not isinstance(vec_b, (np.ndarray, csr_array))
    ):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, either numpy.ndarray "
            f"or scipy.sparse.csr_array, got {type(vec_a)} and {type(vec_b)}"
        )

    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Vectors must have same shape, got {vec_a.shape} and {vec_b.shape}"
        )
