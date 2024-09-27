from typing import Union

import numpy as np
from numba import njit, prange
from scipy.sparse import csr_array
from scipy.spatial.distance import jaccard
from sklearn.utils._param_validation import validate_params


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def tanimoto_binary_similarity(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    """
    Tanimoto similarity for binary vectors.

    Computes the Tanimoto similarity [1]_ for binary data between two input arrays
    or sparse matrices using the Jaccard index. The calculated similarity falls within
    the explicit range [0, 1], with passing all-zero vectors to this function resulting
    in a similarity of 1.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    ----------
    similarity : float
        Tanimoto similarity between vec_a and vec_b.

    References
    ----------
    .. [1] `Bajusz, D., Rácz, A. & Héberger, K.
       "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?"
       J Cheminform, 7, 20 (2015).
       <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3>`_

    Examples
    ----------
    >>> from skfp.similarities import tanimoto_binary_similarity
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = tanimoto_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = tanimoto_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0
    """
    _check_nan(vec_a)
    _check_nan(vec_b)

    if np.sum(vec_a) == 0 == np.sum(vec_b):
        return 1.0

    if isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        vec_a_bool = vec_a.astype(bool)
        vec_b_bool = vec_b.astype(bool)

        intersection: float = vec_a_bool.multiply(vec_b_bool).sum()
        union: float = vec_a_bool.sum() + vec_b_bool.sum() - intersection

        return intersection / union

    elif isinstance(vec_a, np.ndarray) and isinstance(vec_b, np.ndarray):
        vec_a_bool = vec_a.astype(bool)
        vec_b_bool = vec_b.astype(bool)

        return 1 - jaccard(vec_a_bool, vec_b_bool)
    else:
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type: either numpy.ndarray "
            f"or scipy.sparse.csr_array, got {type(vec_a)} and {type(vec_b)}"
        )


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def tanimoto_binary_distance(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    """
    Tanimoto distance for binary vectors.

    Computes the Tanimoto distance for binary data between two input arrays or sparse matrices
    by subtracting the similarity value from 1. The calculated distance falls within
    the range [0, 1], with passing all-zero vectors to this function resulting in a distance of 0.

    Operations on NumPy arrays are optimized with the Numba JIT compiler, making them
    significantly faster than equivalent operations on SciPy csr_arrays.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    ----------
    distance : float
        Tanimoto distance between vec_a and vec_b.

    Examples
    ----------
    >>> from skfp.similarities import tanimoto_binary_distance
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = tanimoto_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = tanimoto_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """

    return 1 - tanimoto_binary_similarity(vec_a, vec_b)


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def tanimoto_count_similarity(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    """
    Computes the Tanimoto similarity [1]_ for count data between two input arrays
    or sparse matrices using formula:

    .. math::

        sim(vec_a, vec_b) = \\frac{vec_a \\cdot vec_b}{\\|vec_a\\|^2 + \\|vec_b\\|^2 - vec_a \\cdot vec_b}

    Calculated similarity falls within the range of 0-1.
    Passing all-zero vectors to this function result in similarity of 1.
    Note that the NumPy implementation is JIT-compiled;
    therefore, it may execute faster than the SciPy implementation.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    ----------
    similarity : float
        Tanimoto similarity between vec_a and vec_b.

    References
    ----------
    .. [1] `Bajusz, D., Rácz, A. & Héberger, K.
       "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?"
       J Cheminform, 7, 20 (2015).
       <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3>`_

    Examples
    ----------
    >>> from skfp.similarities import tanimoto_count_similarity
    >>> vec_a = np.array([7, 1, 1])
    >>> vec_b = np.array([7, 1, 2])
    >>> sim = tanimoto_count_similarity(vec_a, vec_b)
    >>> sim
    0.98

    >>> vec_a = csr_array(([7, 1, 1]))
    >>> vec_b = csr_array(([7, 1, 2]))
    >>> sim = tanimoto_count_similarity(vec_a, vec_b)
    >>> sim
    0.98
    """
    _check_nan(vec_a)
    _check_nan(vec_b)

    if np.sum(vec_a) == 0 and np.sum(vec_b) == 0:
        return 1.0

    if isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        return _tanimoto_count_scipy(vec_a, vec_b)
    elif isinstance(vec_a, np.ndarray) and isinstance(vec_b, np.ndarray):
        return _tanimoto_count_numpy(vec_a, vec_b)
    else:
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type: either numpy.ndarray "
            f"or scipy.sparse.csr_array, got {type(vec_a)} and {type(vec_b)}"
        )


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def tanimoto_count_distance(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    """
    Computes the Tanimoto distance for binary data between two input arrays or sparse matrices
    by subtracting similarity value from 1.
    Calculated distance falls within the range from 0 to 1.
    Passing all-zero vectors to this function results in distance of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    ----------
    distance : float
        Tanimoto distance between vec_a and vec_b.

    Examples
    ----------
    >>> from skfp.similarities import tanimoto_count_distance
    >>> vec_a = np.array([7, 1, 1])
    >>> vec_b = np.array([7, 1, 2])
    >>> dist = tanimoto_count_distance(vec_a, vec_b)
    >>> dist
    0.02

    >>> vec_a = csr_array(([7, 1, 1]))
    >>> vec_b = csr_array(([7, 1, 2]))
    >>> dist = tanimoto_count_distance(vec_a, vec_b)
    >>> dist
    0.02
    """

    return 1 - tanimoto_count_similarity(vec_a, vec_b)


@njit(parallel=True)
def _tanimoto_count_numpy(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculates the Tanimoto similarity between two count data numpy arrays.
    This function uses Numba for JIT compilation and parallel computations to
    efficiently compute the similarity between two vectors.
    """
    vec_a = vec_a.astype(np.float64).ravel()
    vec_b = vec_b.astype(np.float64).ravel()

    dot_ab = 0.0
    dot_aa = 0.0
    dot_bb = 0.0

    for i in prange(vec_a.shape[0]):
        dot_ab += vec_a[i] * vec_b[i]
        dot_aa += vec_a[i] * vec_a[i]
        dot_bb += vec_b[i] * vec_b[i]

    denominator: float = dot_aa + dot_bb - dot_ab

    return dot_ab / denominator


def _tanimoto_count_scipy(vec_a: csr_array, vec_b: csr_array) -> float:
    """
    Calculates the Tanimoto similarity between two count data scipy arrays.
    """
    dot_ab: float = vec_a.multiply(vec_b).sum()
    dot_aa: float = vec_a.multiply(vec_a).sum()
    dot_bb: float = vec_b.multiply(vec_b).sum()

    denominator: float = dot_aa + dot_bb - dot_ab

    tanimoto_sim: float = dot_ab / denominator

    return tanimoto_sim


def _check_nan(arr: Union[np.ndarray, csr_array]) -> None:
    """
    Checks if passed numpy array or scipy sparse matrix contains NaN values.
    """
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
