from typing import Union

import numpy as np
from numba import njit, prange
from scipy.sparse import csr_array
from sklearn.utils._param_validation import validate_params

from .utils import _check_nan


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def all_bit_binary_similarity(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    r"""
    Calculate the all-bit binary similarity between two binary vectors.

    Computes the all-bit similarity for binary data between two input arrays
    or sparse matrices using the formula:

    .. math::

        sim(vec_a, vec_b) = \frac{\text{count}_a - \text{count}_\text{xor}}{\text{count}_a}

    Where:
    - \( \text{count}_a \) is the number of non-zero elements in \( \mathbf{a} \),
    - \( \text{count}_\text{xor} \) is the number of positions where the binary values in \( \mathbf{a} \) and \( \mathbf{b} \) differ.

    The calculated similarity falls within the range ``[0, 1]``.
    Passing all-zero vectors to this function results in a similarity of 1.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        All-Bit similarity between vec_a and vec_b.

    References
    ----------
    TODO.

    Examples
    --------
    >>> from skfp.distances import all_bit_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = all_bit_binary_similarity(vec_a, vec_b)
    >>> sim  # doctest: +SKIP
    1.0

    >>> from skfp.distances import all_bit_binary_similarity
    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = all_bit_binary_similarity(vec_a, vec_b)
    >>> sim  # doctest: +SKIP
    1.0
    """
    _check_nan(vec_a)
    _check_nan(vec_b)

    if np.sum(vec_a) == 0 == np.sum(vec_b):
        return 1.0

    if isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        vec_a_bool = vec_a.astype(bool)
        vec_b_bool = vec_b.astype(bool)
        return _all_bit_binary_scipy(vec_a_bool, vec_b_bool)

    elif isinstance(vec_a, np.ndarray) and isinstance(vec_b, np.ndarray):
        vec_a_bool = vec_a.astype(bool)
        vec_b_bool = vec_b.astype(bool)
        return _all_bit_binary_numpy(vec_a_bool, vec_b_bool)

    else:
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, either numpy.ndarray "
            f"or scipy.sparse.csr_array, got {type(vec_a)} and {type(vec_b)}"
        )


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def all_bit_binary_distance(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    """
    All-Bit distance for vectors of binary values.

    Computes the All-Bit distance for binary data between two input arrays
    or sparse matrices by subtracting the similarity from 1, using to
    the formula:

    .. math::

        dist(vec_a, vec_b) = 1 - sim(vec_a, vec_b)

    The calculated distance falls within the range ``[0, 1]``.
    Passing all-zero vectors to this function results in a distance of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    distance : float
        All-Bit distance between ``vec_a`` and ``vec_b``.

    Examples
    --------
    >>> from skfp.distances import all_bit_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = all_bit_binary_distance(vec_a, vec_b)
    >>> dist  # doctest: +SKIP
    0.0

    >>> from skfp.distances import all_bit_binary_distance
    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = all_bit_binary_distance(vec_a, vec_b)
    >>> dist  # doctest: +SKIP
    0.0
    """
    return 1 - all_bit_binary_similarity(vec_a, vec_b)


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def all_bit_count_similarity(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    r"""
    All-bit similarity for vectors of count values.

    Computes the all-bit for count data between two input arrays
    or sparse matrices using the formula:

    .. math::

        sim(vec_a, vec_b) = \frac{\text{count}_a - \text{count}_\text{diff}}{\text{count}_a}

    Where:
    - \( \text{count}_a \) is the total sum of values in \( \mathbf{a} \),
    - \( \text{count}_\text{diff} \) is the absolute difference of values at the same positions in \( \mathbf{a} \) and \( \mathbf{b} \).

    The calculated similarity falls within the range ``[0, 1]``.
    Passing all-zero vectors to this function results in a similarity of 1

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First count input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second count input array or sparse matrix.

    Returns
    -------
    similarity : float
        All-Bit similarity between vec_a and vec_b.

    References
    ----------
    TODO.

    Examples
    --------
    >>> from skfp.distances import all_bit_count_similarity
    >>> import numpy as np
    >>> vec_a = np.array([7, 1, 1])
    >>> vec_b = np.array([7, 1, 1])
    >>> sim = all_bit_count_similarity(vec_a, vec_b)
    >>> sim  # doctest: +SKIP
    1.0

    >>> from skfp.distances import all_bit_count_similarity
    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[7, 1, 1]])
    >>> vec_b = csr_array([[7, 1, 1]])
    >>> sim = all_bit_count_similarity(vec_a, vec_b)
    >>> sim  # doctest: +SKIP
    1.0
    """
    _check_nan(vec_a)
    _check_nan(vec_b)

    if np.sum(vec_a) == 0 and np.sum(vec_b) == 0:
        return 1.0

    if isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        return _all_bit_count_scipy(vec_a, vec_b)
    elif isinstance(vec_a, np.ndarray) and isinstance(vec_b, np.ndarray):
        return _all_bit_count_numpy(vec_a, vec_b)
    else:
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, either numpy.ndarray "
            f"or scipy.sparse.csr_array, got {type(vec_a)} and {type(vec_b)}"
        )


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def all_bit_count_distance(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    """
    All-Bit distance for vectors of count values.

    Computes the All-Bit distance for count data between two input arrays
    or sparse matrices by subtracting the similarity from 1, using to
    the formula:

    .. math::

        dist(vec_a, vec_b) = 1 - sim(vec_a, vec_b)

    The calculated distance falls within the range ``[0, 1]``.
    Passing all-zero vectors to this function results in a distance of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First count input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second count input array or sparse matrix.

    Returns
    -------
    distance : float
        All-Bit distance between vec_a and vec_b.

    Examples
    --------
    >>> from skfp.distances import all_bit_count_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = all_bit_count_distance(vec_a, vec_b)
    >>> dist  # doctest: +SKIP
    0.0

    >>> from skfp.distances import all_bit_count_distance
    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = all_bit_count_distance(vec_a, vec_b)
    >>> dist  # doctest: +SKIP
    0.0
    """
    return 1 - all_bit_count_similarity(vec_a, vec_b)


@njit(parallel=True)
def _all_bit_binary_numpy(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    count_a = np.count_nonzero(vec_a)
    xor_result = np.empty_like(vec_a, dtype=vec_a.dtype)

    for i in prange(vec_a.size):
        xor_result[i] = vec_a[i] ^ vec_b[i]

    count_xor = np.count_nonzero(xor_result)

    if count_a == 0:
        return 0.0

    all_bit_sim = (count_a - count_xor) / count_a
    return max(0.0, all_bit_sim)


def _all_bit_binary_scipy(vec_a: csr_array, vec_b: csr_array) -> float:
    a_idxs = set(vec_a.indices)
    b_idxs = set(vec_b.indices)

    xor_count = len(a_idxs.symmetric_difference(b_idxs))

    count_a = len(a_idxs)

    if count_a == 0:
        return 0.0

    all_bit_sim = (count_a - xor_count) / count_a
    return max(0.0, all_bit_sim)


@njit(parallel=True)
def _all_bit_count_numpy(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    count_a = np.sum(vec_a)
    count_diff = 0
    for i in prange(vec_a.size):
        count_diff += abs(vec_a[i] - vec_b[i])

    if count_a == 0:
        return 0.0

    all_bit_sim = (count_a - count_diff) / count_a
    return max(0.0, all_bit_sim)


def _all_bit_count_scipy(vec_a: csr_array, vec_b: csr_array) -> float:
    a_data = vec_a.data
    a_indices = vec_a.indices
    b_data = vec_b.data
    b_indices = vec_b.indices

    count_a = a_data.sum()
    count_diff = 0

    for i, idx in enumerate(a_indices):
        a_val = a_data[i]
        b_val = 0
        if idx in b_indices:
            b_val = b_data[b_indices.searchsorted(idx)]

        count_diff += abs(a_val - b_val)

    for i, idx in enumerate(b_indices):
        if idx not in a_indices:
            count_diff += abs(b_data[i])

    if count_a == 0:
        return 0.0

    all_bit_sim = (count_a - count_diff) / count_a
    return max(0.0, all_bit_sim)
