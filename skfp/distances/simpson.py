from typing import Optional, Union

import numba
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
def simpson_binary_similarity(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    r"""
    Simpson similarity for vectors of binary values.

    Computes the Simpson similarity [1]_ (also known as asymmetric similarity [2]_ [3]_
    or overlap coefficient [4]_) for binary data between two input arrays or sparse
    matrices using the formula:

    .. math::

        sim(a, b) = \frac{|a \cap b|}{\min(|a|, |b|)}

    The calculated similarity falls within the range :math:`[0, 1]`.
    If any of the vectors is all-zeros, it results in a similarity of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        Simpson similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Simpson, G.G.
        "Mammals and the nature of continents."
        American Journal of Science, 241: 1-31 (1943).
        <https://doi.org/10.1038/163688a0>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    .. [4] `Overlap coefficient on Wikipedia
        <https://en.wikipedia.org/wiki/Overlap_coefficient>`_

    Examples
    --------
    >>> from skfp.distances import simpson_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = simpson_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = simpson_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if isinstance(vec_a, np.ndarray):
        num_common = np.sum(np.logical_and(vec_a, vec_b))
    else:
        num_common = len(set(vec_a.indices) & set(vec_b.indices))

    min_vec = min(np.sum(vec_a), np.sum(vec_b))

    sim = num_common / min_vec if min_vec != 0 else 0.0
    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def simpson_binary_distance(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    """
    Simpson distance for vectors of binary values.

    Computes the Simpson distance for binary data between two input arrays
    or sparse matrices by subtracting the Simpson similarity [1]_ [2]_ [3]_ [4]_
    from 1, using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`simpson_binary_similarity`.
    The calculated distance falls within the range :math:`[0, 1]`.
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
        Simpson distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Simpson, G.G.
        "Mammals and the nature of continents."
        American Journal of Science, 241: 1-31 (1943).
        <https://doi.org/10.1038/163688a0>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    .. [4] `Overlap coefficient on Wikipedia
        <https://en.wikipedia.org/wiki/Overlap_coefficient>`_

    Examples
    --------
    >>> from skfp.distances import simpson_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = simpson_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = simpson_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - simpson_binary_similarity(vec_a, vec_b)


@validate_params(
    {"X": ["array-like"], "Y": ["array-like", None]},
    prefer_skip_nested_validation=True,
)
def bulk_simpson_binary_similarity(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    Bulk Simpson similarity for binary matrices.

    Computes the pairwise Simpson 1]_ (also known as asymmetric similarity [2]_ [3]_
    or overlap coefficient [4]_) similarity between binary matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`simpson_binary_similarity`.

    Parameters
    ----------
    X : ndarray
        First binary input array, of shape :math:`m \times m`

    Y : ndarray, default=None
        Second binary input array, of shape :math:`n \times n`. If not passed, similarities
        are computed between rows of X.

    Returns
    -------
    similarities : ndarray
        Array with pairwise Simpson similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    References
    ----------
    .. [1] `Simpson, G.G.
        "Mammals and the nature of continents."
        American Journal of Science, 241: 1-31 (1943).
        <https://doi.org/10.1038/163688a0>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    .. [4] `Overlap coefficient on Wikipedia
        <https://en.wikipedia.org/wiki/Overlap_coefficient>`_

    See Also
    --------
    :py:func:`simpson_binary_similarity` : Simpson similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_simpson_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_simpson_binary_similarity(X, Y)
    >>> sim
    array([[1. , 0.5],
           [1. , 1. ]])
    """
    if Y is None:
        return _bulk_simpson_binary_similarity_single(X)
    else:
        return _bulk_simpson_binary_similarity_two(X, Y)


@numba.njit(parallel=True)
def _bulk_simpson_binary_similarity_single(X: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    sims = np.empty((m, m))
    row_sums = np.sum(X, axis=1)

    # upper triangle - actual similarities
    for i in numba.prange(m):
        for j in numba.prange(i + 1, m):
            num_common = np.sum(np.logical_and(X[i], X[j]))
            min_vec = min(row_sums[i], row_sums[j])
            sims[i, j] = num_common / min_vec if min_vec != 0 else 0.0

    # diagonal - always 1
    for i in numba.prange(m):
        sims[i, i] = 1.0

    # lower triangle - symmetric with upper triangle
    for i in numba.prange(1, m):
        for j in numba.prange(i):
            sims[i, j] = sims[j, i]

    return sims


@numba.njit(parallel=True)
def _bulk_simpson_binary_similarity_two(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    n = Y.shape[0]
    sims = np.empty((m, n))

    row_sums_X = np.sum(X, axis=1)
    row_sums_Y = np.sum(Y, axis=1)

    for i in numba.prange(m):
        for j in numba.prange(m):
            num_common = np.sum(np.logical_and(X[i], Y[j]))
            min_vec = min(row_sums_X[i], row_sums_Y[j])
            sims[i, j] = num_common / min_vec if min_vec != 0 else 0.0

    return sims


@validate_params(
    {
        "X": ["array-like", csr_array],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_simpson_binary_distance(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    Bulk Simpson distance for vectors of binary values.

    Computes the pairwise Simpson distance between binary matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`simpson_binary_distance`.

    Parameters
    ----------
    X : ndarray
        First binary input array, of shape :math:`m \times m`

    Y : ndarray, default=None
        Second binary input array, of shape :math:`n \times n`. If not passed, distances
        are computed between rows of X.

    Returns
    -------
    distances : ndarray
        Array with pairwise Simpson distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`simpson_binary_distance` : Simpson distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_simpson_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_simpson_binary_distance(X, Y)
    >>> dist
    array([[0., 0.],
           [0., 0.]])

    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_simpson_binary_distance(X)
    >>> dist
    array([[0., 0.],
           [0., 0.]])
    """
    return 1 - bulk_simpson_binary_similarity(X, Y)
