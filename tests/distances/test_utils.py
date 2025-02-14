import numpy as np
import pytest
from scipy.sparse import csr_array

from skfp.distances.utils import _check_finite_values


def test_check_nan_numpy():
    vec_a = np.array([1, 2, np.nan, 4, 5])

    with pytest.raises(ValueError, match="Input array contains infinity or NaN values"):
        _check_finite_values(vec_a)


def test_check_inf_numpy():
    vec_a = np.array([1, 2, np.inf, 4, 5])

    with pytest.raises(ValueError, match="Input array contains infinity or NaN values"):
        _check_finite_values(vec_a)


def test_check_nan_scipy():
    sparse_matrix = csr_array([[1, 2, np.nan, 4, 5]])

    with pytest.raises(
        ValueError, match="Input sparse matrix contains infinity or NaN values"
    ):
        _check_finite_values(sparse_matrix)


def test_check_nan_wrong_type():
    with pytest.raises(TypeError) as exc_info:
        _check_finite_values(1)

    assert "Expected numpy.ndarray or scipy.sparse.csr_array" in str(exc_info)
