from typing import Union

import numpy as np
from numba import njit, prange
from scipy.sparse import csr_array
from scipy.spatial.distance import jaccard
from sklearn.utils._param_validation import validate_params

from .utils import _check_nan


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
    r"""
    Tanimoto similarity for vectors of binary values.

    Computes the Tanimoto similarity [1]_ for binary data between two input arrays
    or sparse matrices using the Jaccard index using formula:

    .. math::

        sim(vec_a, vec_b) = \frac{|vec_a \cap vec_b|}{|vec_a| + |vec_b| - |vec_a \cap vec_b|}

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
        Tanimoto similarity between vec_a and vec_b.

    References
    ----------
    .. [1] `Bajusz, D., Rácz, A. & Héberger, K.
       "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?"
       J Cheminform, 7, 20 (2015).
       <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3>`_

    Examples
    --------
    >>> from skfp.distances import tanimoto_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = tanimoto_binary_similarity(vec_a, vec_b)
    >>> sim  # doctest: +SKIP
    1.0

    >>> from skfp.distances import tanimoto_binary_similarity
    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = tanimoto_binary_similarity(vec_a, vec_b)
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

        intersection: float = vec_a_bool.multiply(vec_b_bool).sum()
        union: float = vec_a_bool.sum() + vec_b_bool.sum() - intersection

        return intersection / union

    elif isinstance(vec_a, np.ndarray) and isinstance(vec_b, np.ndarray):
        vec_a_bool = vec_a.astype(bool)
        vec_b_bool = vec_b.astype(bool)

        return 1 - jaccard(vec_a_bool, vec_b_bool)
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
def tanimoto_binary_distance(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    """
    Tanimoto distance for vectors of binary values.

    Computes the Tanimoto distance for binary data between two input arrays
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
        Tanimoto distance between ``vec_a`` and ``vec_b``.

    Examples
    --------
    >>> from skfp.distances import tanimoto_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = tanimoto_binary_distance(vec_a, vec_b)
    >>> dist  # doctest: +SKIP
    0.0

    >>> from skfp.distances import tanimoto_binary_distance
    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = tanimoto_binary_distance(vec_a, vec_b)
    >>> dist  # doctest: +SKIP
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
    r"""
    Tanimoto similarity for vectors of count values.

    Computes the Tanimoto similarity [1]_ for count data between two input arrays
    or sparse matrices using the formula:

    .. math::

        sim(vec_a, vec_b) = \frac{vec_a \cdot vec_b}{\|vec_a\|^2 + \|vec_b\|^2 - vec_a \cdot vec_b}

    Calculated similarity falls within the range of `[0, 1]`.
    Passing all-zero vectors to this function results in similarity of 1.

    Note that Numpy version is optimized with Numba JIT compiler, resulting in significantly faster
    performance compared to SciPy sparse arrays. First usage may be slightly slower due to Numba compilation.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First count input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second count input array or sparse matrix.

    Returns
    -------
    similarity : float
        Tanimoto similarity between vec_a and vec_b.

    References
    ----------
    .. [1] `Bajusz, D., Rácz, A. & Héberger, K.
       "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?"
       J Cheminform, 7, 20 (2015).
       <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3>`_

    Examples
    --------
    >>> from skfp.distances import tanimoto_count_similarity
    >>> import numpy as np
    >>> vec_a = np.array([7, 1, 1])
    >>> vec_b = np.array([7, 1, 2])
    >>> sim = tanimoto_count_similarity(vec_a, vec_b)
    >>> sim  # doctest: +SKIP
    0.9811320754716981

    >>> from skfp.distances import tanimoto_count_similarity
    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[7, 1, 1]])
    >>> vec_b = csr_array([[7, 1, 2]])
    >>> sim = tanimoto_count_similarity(vec_a, vec_b)
    >>> sim  # doctest: +SKIP
    0.9811320754716981
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
def tanimoto_count_distance(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    """
    Tanimoto distance for vectors of count values.

    Computes the Tanimoto distance for binary data between two input arrays
    or sparse matrices by subtracting similarity value from 1, using to
    the formula:

    .. math::

            dist(vec_a, vec_b) = 1 - sim(vec_a, vec_b)

    Calculated distance falls within the range from `[0, 1]`.
    Passing all-zero vectors to this function results in distance of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First count input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second count input array or sparse matrix.

    Returns
    -------
    distance : float
        Tanimoto distance between vec_a and vec_b.

    Examples
    --------
    >>> from skfp.distances import tanimoto_count_distance
    >>> import numpy as np
    >>> vec_a = np.array([7, 1, 1])
    >>> vec_b = np.array([7, 1, 2])
    >>> dist = tanimoto_count_distance(vec_a, vec_b)
    >>> dist  # doctest: +SKIP
    0.018867924528301883

    >>> from skfp.distances import tanimoto_count_distance
    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[7, 1, 1]])
    >>> vec_b = csr_array([[7, 1, 2]])
    >>> dist = tanimoto_count_distance(vec_a, vec_b)
    >>> dist  # doctest: +SKIP
    0.018867924528301883
    """
    return 1 - tanimoto_count_similarity(vec_a, vec_b)


@njit(parallel=True)
def _tanimoto_count_numpy(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    vec_a = vec_a.astype(np.float64).ravel()
    vec_b = vec_b.astype(np.float64).ravel()

    dot_ab = 0.0
    dot_aa = 0.0
    dot_bb = 0.0

    for i in prange(vec_a.shape[0]):
        dot_ab += vec_a[i] * vec_b[i]
        dot_aa += vec_a[i] * vec_a[i]
        dot_bb += vec_b[i] * vec_b[i]

    return dot_ab / (dot_aa + dot_bb - dot_ab)


def _tanimoto_count_scipy(vec_a: csr_array, vec_b: csr_array) -> float:
    dot_ab: float = vec_a.multiply(vec_b).sum()
    dot_aa: float = vec_a.multiply(vec_a).sum()
    dot_bb: float = vec_b.multiply(vec_b).sum()

    return dot_ab / (dot_aa + dot_bb - dot_ab)
