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
def tanimoto_binary_similarity(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    r"""
    Tanimoto similarity for vectors of binary values.

    Computes the Tanimoto similarity [1]_ (also known as Jaccard similarity)
    for binary data between two input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{|a \cap b|}{|a \cup b|} = \frac{|a \cap b|}{|a| + |b| - |a \cap b|}

    The calculated similarity falls within the range :math:`[0, 1]`.
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
        Tanimoto similarity between ``vec_a`` and ``vec_b``.

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
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = tanimoto_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if isinstance(vec_a, np.ndarray):
        intersection = np.sum(np.logical_and(vec_a, vec_b))
        union = np.sum(np.logical_or(vec_a, vec_b))
    else:
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)
        intersection = len(vec_a_idxs & vec_b_idxs)
        union = len(vec_a_idxs | vec_b_idxs)

    sim = intersection / union if union != 0 else 1.0
    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def tanimoto_binary_distance(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    """
    Tanimoto distance for vectors of binary values.

    Computes the Tanimoto distance [1]_ (also known as Jaccard distance)
    for binary data between two input arrays or sparse matrices by subtracting
    the similarity from 1, using the formula:

    .. math::

        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`tanimoto_binary_similarity`.
    The calculated distance falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a distance of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    References
    ----------
    .. [1] `Bajusz, D., Rácz, A. & Héberger, K.
        "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?"
        J Cheminform, 7, 20 (2015).
        <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3>`_

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
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
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
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    r"""
    Tanimoto similarity for vectors of count values.

    Computes the Tanimoto similarity [1]_ for count data between two input arrays
    or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{a \cdot b}{\|a\|^2 + \|b\|^2 - a \cdot b}

    Calculated similarity falls within the range of :math:`[0, 1]`.
    Passing all-zero vectors to this function results in similarity of 1.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First count input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second count input array or sparse matrix.

    Returns
    -------
    similarity : float
        Tanimoto similarity between ``vec_a`` and ``vec_b``.

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
    >>> sim
    0.9811320754716981

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[7, 1, 1]])
    >>> vec_b = csr_array([[7, 1, 2]])
    >>> sim = tanimoto_count_similarity(vec_a, vec_b)
    >>> sim
    0.9811320754716981
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if isinstance(vec_a, np.ndarray):
        dot_aa = np.dot(vec_a, vec_a)
        dot_bb = np.dot(vec_b, vec_b)
        dot_ab = np.dot(vec_a, vec_b)
    else:
        dot_ab = vec_a.multiply(vec_b).sum()
        dot_aa = vec_a.multiply(vec_a).sum()
        dot_bb = vec_b.multiply(vec_b).sum()

    intersection = dot_ab
    union = dot_aa + dot_bb - dot_ab

    sim = intersection / union if union >= 1e-8 else 1.0
    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def tanimoto_count_distance(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    """
    Tanimoto distance for vectors of count values.

    Computes the Tanimoto distance [1]_ for binary data between two input arrays
    or sparse matrices by subtracting similarity value from 1, using the formula:

    .. math::

            dist(a, b) = 1 - sim(a, b)

    See also :py:func:`tanimoto_count_similarity`.
    Calculated distance falls within the range from :math:`[0, 1]`.
    Passing all-zero vectors to this function results in distance of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First count input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second count input array or sparse matrix.

    References
    ----------
    .. [1] `Bajusz, D., Rácz, A. & Héberger, K.
        "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?"
        J Cheminform, 7, 20 (2015).
        <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3>`_

    Returns
    -------
    distance : float
        Tanimoto distance between ``vec_a`` and ``vec_b``.

    Examples
    --------
    >>> from skfp.distances import tanimoto_count_distance
    >>> import numpy as np
    >>> vec_a = np.array([7, 1, 1])
    >>> vec_b = np.array([7, 1, 2])
    >>> dist = tanimoto_count_distance(vec_a, vec_b)
    >>> dist
    0.018867924528301883

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[7, 1, 1]])
    >>> vec_b = csr_array([[7, 1, 2]])
    >>> dist = tanimoto_count_distance(vec_a, vec_b)
    >>> dist
    0.018867924528301883
    """
    return 1 - tanimoto_count_similarity(vec_a, vec_b)


@validate_params(
    {"vec_a": ["array-like"], "vec_b": ["array-like"]},
    prefer_skip_nested_validation=True,
)
def bulk_tanimoto_binary_similarity(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    Bulk Tanimoto similarity for binary matrices.

    Computes the pairwise Tanimoto [1]_ similarity between binary matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`tanimoto_binary_similarity`.

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
        Array with pairwise Tanimoto similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    References
    ----------
    .. [1] `Bajusz, D., Rácz, A. & Héberger, K.
        "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?"
        J Cheminform, 7, 20 (2015).
        <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3>`_

    See Also
    --------
    :py:func:`tanimoto_binary_similarity` : Tanimoto similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_tanimoto_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([[1, 0, 1], [0, 0, 1]])
    >>> vec_b = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_tanimoto_binary_similarity(vec_a, vec_b)
    >>> sim
    [[1.0, 1.0], [1.0, 1.0]]
    """
    if Y is None:
        return _bulk_tanimoto_binary_similarity_single(X)
    else:
        return _bulk_tanimoto_binary_similarity_two(X, Y)


@numba.njit(parallel=True)
def _bulk_tanimoto_binary_similarity_single(X: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    sims = np.empty((m, m))

    # upper triangle - actual similarities
    for i in numba.prange(m):
        for j in numba.prange(i + 1, m):
            intersection = np.sum(np.logical_and(X[i], X[j]))
            union = np.sum(np.logical_or(X[i], X[j]))
            sims[i, j] = intersection / union if union != 0 else 1.0

    # diagonal - always 1
    for i in numba.prange(m):
        sims[i, i] = 1.0

    # lower triangle - symmetric with upper triangle
    for i in numba.prange(1, m):
        for j in numba.prange(i):
            sims[i, j] = sims[j, i]

    return sims


@numba.njit(parallel=True)
def _bulk_tanimoto_binary_similarity_two(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    n = Y.shape[0]
    sims = np.empty((m, n))

    for i in numba.prange(m):
        for j in numba.prange(m):
            intersection = np.sum(np.logical_and(X[i], Y[j]))
            union = np.sum(np.logical_or(X[i], X[j]))
            sims[i, j] = intersection / union if union != 0 else 1.0

    return sims


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def bulk_tanimoto_binary_distance(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> float:
    r"""
    Bulk Tanimoto distance for vectors of binary values.

    Computes the pairwise Tanimoto distance between binary matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`tanimoto_binary_distance`.

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
        Array with pairwise Tanimoto distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`tanimoto_binary_distance` : Tanimoto distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_tanimoto_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_tanimoto_binary_distance(X, Y)
    >>> dist
    [[1, 1], [1, 1]]

    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_tanimoto_binary_distance(X)
    >>> dist
    [[1, 1], [1, 1]]
    """
    return 1 - bulk_tanimoto_binary_similarity(X, Y)


@validate_params(
    {"vec_a": ["array-like"], "vec_b": ["array-like"]},
    prefer_skip_nested_validation=True,
)
def bulk_tanimoto_count_similarity(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    Bulk Tanimoto similarity for count matrices.

    Computes the pairwise Tanimoto similarity between count matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`tanimoto_count_similarity`.

    Parameters
    ----------
    X : ndarray
        First count input array, of shape :math:`m \times m`

    Y : ndarray, default=None
        Second count input array, of shape :math:`n \times n`. If not passed, similarities
        are computed between rows of X.

    Returns
    -------
    similarities : ndarray
        Array with pairwise Tanimoto similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`tanimoto_count_similarity` : Tanimoto similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_tanimoto_count_similarity
    >>> import numpy as np
    >>> vec_a = np.array([[1, 0, 1], [0, 0, 1]])
    >>> vec_b = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_tanimoto_count_similarity(vec_a, vec_b)
    >>> sim
    [[1.0, 0.5], [0.5, 1.0]]
    """
    if Y is None:
        return _bulk_tanimoto_count_similarity_single(X)
    else:
        return _bulk_tanimoto_count_similarity_two(X, Y)


@numba.njit(parallel=True)
def _bulk_tanimoto_count_similarity_single(X: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    sims = np.empty((m, m))

    # upper triangle - actual similarities
    for i in numba.prange(m):
        for j in numba.prange(i + 1, m):
            vec_a = X[i]
            vec_b = X[j]

            dot_aa = np.dot(vec_a, vec_a)
            dot_bb = np.dot(vec_b, vec_b)
            dot_ab = np.dot(vec_a, vec_b)

            intersection = dot_ab
            union = dot_aa + dot_bb - dot_ab

            sims[i, j] = intersection / union if union >= 1e-8 else 1.0

    # diagonal - always 1
    for i in numba.prange(m):
        sims[i, i] = 1.0

    # lower triangle - symmetric with upper triangle
    for i in numba.prange(1, m):
        for j in numba.prange(i):
            sims[i, j] = sims[j, i]

    return sims


@numba.jit(parallel=True)
def _bulk_tanimoto_count_similarity_two(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    n = Y.shape[0]
    sims = np.empty((m, n))

    for i in numba.prange(m):
        for j in numba.prange(n):
            vec_a = X[i]
            vec_b = X[j]

            dot_aa = np.dot(vec_a, vec_a)
            dot_bb = np.dot(vec_b, vec_b)
            dot_ab = np.dot(vec_a, vec_b)

            intersection = dot_ab
            union = dot_aa + dot_bb - dot_ab

            sims[i, j] = intersection / union if union >= 1e-8 else 1.0

    return sims


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def bulk_tanimoto_count_distance(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> float:
    r"""
    Bulk Tanimoto distance for vectors of count values.

    Computes the pairwise Tanimoto [1]_ distance between count matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`tanimoto_count_distance`.

    Parameters
    ----------
    X : ndarray
        First count input array, of shape :math:`m \times m`

    Y : ndarray, default=None
        Second count input array, of shape :math:`n \times n`. If not passed, distances
        are computed between rows of X.

    Returns
    -------
    distances : ndarray
        Array with pairwise Tanimoto distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    References
    ----------
    .. [1] `Bajusz, D., Rácz, A. & Héberger, K.
        "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?"
        J Cheminform, 7, 20 (2015).
        <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3>`_

    See Also
    --------
    :py:func:`tanimoto_count_distance` : Tanimoto distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_tanimoto_count_distance
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_tanimoto_count_distance(X, Y)
    >>> dist
    [[0.0, 0.0], [0.0, 0.0]]

    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_tanimoto_count_distance(X)
    >>> dist
    [[0.0, 0.0], [0.0, 0.0]]
    """
    return 1 - bulk_tanimoto_binary_similarity(X, Y)
