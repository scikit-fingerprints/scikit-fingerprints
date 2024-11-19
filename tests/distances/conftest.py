import numpy as np
import pytest
from scipy.sparse import csr_array


@pytest.fixture
def binary_numpy_array():
    vec = np.array([1, 0, 1, 0, 1])

    return vec


@pytest.fixture
def binary_csr_array():
    vec = csr_array([[1, 0, 1, 0, 1]])

    return vec
