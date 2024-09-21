from typing import Union

import numpy as np
from scipy.sparse import csr_array
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
    Computes the Tanimoto similarity [1]_ for binary data between two input arrays
    or sparse matrices using formula:

    .. math::

        sim(vec_a, vec_b) = \\frac{|vec_a \\cap vec_b|}{|vec_a| + |vec_b| - |vec_a \\cap vec_b|}

    Calculated similarity falls within the range of 0-1.
    Passing all-zero vectors to this function results in similarity of 1.

    Tanimoto similarity is calculated using the formula:

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    ----------
    similarity : float
        Tanimoto similarity between A and vec_b.

    References
    ----------
    .. [1] `Bajusz, D., Rácz, A. & Héberger, K.
    "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?"
    Journal of Cheminformatics, 7, 20 (2015).
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3`_

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

    if isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        return _tanimoto_binary_scipy(vec_a, vec_b)
    elif isinstance(vec_a, np.ndarray) and isinstance(vec_b, np.ndarray):
        return _tanimoto_binary_numpy(vec_a, vec_b)
    else:
        raise TypeError(
            f"Both A and B must be of the same type: either numpy.ndarray "
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
    Computes the Tanimoto distance for binary data between two input arrays or sparse matrices
    by subtracting similarity value from 1.
    Calculated distance falls within the range of 0-1.
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
        Tanimoto distance between A and vec_b.

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

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    ----------
    similarity : float
        Tanimoto similarity between A and vec_b.

    References
    ----------
    .. [1] `Bajusz, D., Rácz, A. & Héberger, K.
    "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?"
    Journal of Cheminformatics, 7, 20 (2015).
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3`_

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

    if isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        return _tanimoto_count_scipy(vec_a, vec_b)
    elif isinstance(vec_a, np.ndarray) and isinstance(vec_b, np.ndarray):
        return _tanimoto_count_numpy(vec_a, vec_b)
    else:
        raise TypeError(
            f"Both A and B must be of the same type: either numpy.ndarray "
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
        Tanimoto distance between A and vec_b.

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


def _tanimoto_binary_numpy(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculates the Tanimoto similarity between two binary numpy arrays.
    """
    _check_nan(vec_a)
    _check_nan(vec_b)

    if vec_a.sum() == 0 == vec_b.sum():
        return 1.0

    dot_ab: float = np.dot(vec_a, vec_b)
    dot_aa: float = np.sum(vec_a)
    dot_bb: float = np.sum(vec_b)

    denominator: float = dot_aa + dot_bb - dot_ab
    _check_zero_denominator(denominator)

    tanimoto_similarity = dot_ab / denominator
    return tanimoto_similarity


def _tanimoto_binary_scipy(vec_a: csr_array, vec_b: csr_array) -> float:
    """
    Calculates the Tanimoto similarity between two binary scipy arrays.
    """
    _check_nan(vec_a)
    _check_nan(vec_b)

    if np.sum(vec_a) == 0 and np.sum(vec_b) == 0:
        return 1.0

    a_and_b: float = vec_a.multiply(vec_b).sum()
    a_sum: float = vec_a.sum()
    b_sum: float = vec_b.sum()

    denominator: float = a_sum + b_sum - a_and_b
    _check_zero_denominator(denominator)

    tanimoto_sim: float = a_and_b / denominator

    return tanimoto_sim


def _tanimoto_count_numpy(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculates the Tanimoto similarity between two count data numpy arrays.
    """
    _check_nan(vec_a)
    _check_nan(vec_b)

    if np.sum(vec_a) == 0 and np.sum(vec_b) == 0:
        return 1.0

    dot_ab: float = np.dot(vec_a, vec_b)
    dot_aa: float = np.dot(vec_a, vec_a)
    dot_bb: float = np.dot(vec_b, vec_b)

    denominator: float = dot_aa + dot_bb - dot_ab
    _check_zero_denominator(denominator)

    tanimoto_sim: float = dot_ab / denominator

    return tanimoto_sim


def _tanimoto_count_scipy(vec_a: csr_array, vec_b: csr_array) -> float:
    """
    Calculates the Tanimoto similarity between two count data scipy arrays.
    """
    _check_nan(vec_a)
    _check_nan(vec_b)

    if np.sum(vec_a) == 0 and np.sum(vec_b) == 0:
        return 1.0

    dot_ab: float = vec_a.multiply(vec_b).sum()
    dot_aa: float = vec_a.multiply(vec_a).sum()
    dot_bb: float = vec_b.multiply(vec_b).sum()

    denominator: float = dot_aa + dot_bb - dot_ab
    _check_zero_denominator(denominator)

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
            "Unsupported type provided. Expected numpy.ndarray or scipy.sparse.csr_array."
        )


def _check_zero_denominator(denominator: float) -> None:
    """
    Checks if passed expression is 0.0 to prevent zero division while calculating Tanimoto.
    """
    if denominator == 0.0:
        raise ZeroDivisionError("Denominator is zero")
