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
def mcconnaughey_binary_similarity(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
    normalized: bool = False,
) -> float:
    r"""
    McConnaughey similarity for vectors of binary values.

    Computes the McConnaughey similarity [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{(|a \cap b| \cdot (|a| + |b|) - |a| \cdot |b|}{|a| \cdot |b|}
                  = \frac{|a \cap b|}{|a|} + \frac{|a \cap b|}{|b|} - 1

    The calculated similarity falls within the range :math:`[-1, 1]`.
    Use ``normalized`` argument to scale the similarity to range :math:`[0, 1]`.
    Passing two all-zero vectors to this function results in a similarity of 1. Passing
    only one all-zero vector results in a similarity of -1 for the non-normalized variant
    and 0 for the normalized variant.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    normalized : bool, default=False
        Whether to normalize values to range ``[0, 1]`` by adding one and dividing the result
        by 2.

    Returns
    -------
    similarity : float
        McConnaughey similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `McConnaughey B.H.
        "The determination and analysis of plankton communities"
        Lembaga Penelitian Laut, 1964.
        <https://books.google.pl/books?id=7aBbOQAACAAJ>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import mcconnaughey_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = mcconnaughey_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = mcconnaughey_binary_similarity(vec_a, vec_b)
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
        vec_a_ones = np.sum(vec_a)
        vec_b_ones = np.sum(vec_b)
    else:
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)

        num_common = len(vec_a_idxs & vec_b_idxs)
        vec_a_ones = len(vec_a_idxs)
        vec_b_ones = len(vec_b_idxs)

    sum_ab_ones = vec_a_ones + vec_b_ones
    dot_ab_ones = vec_a_ones * vec_b_ones

    if sum_ab_ones == 0:
        sim = 1.0
    elif dot_ab_ones == 0:
        sim = -1.0
    else:
        sim = (num_common * sum_ab_ones - dot_ab_ones) / dot_ab_ones

    if normalized:
        sim = (sim + 1) / 2

    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def mcconnaughey_binary_distance(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    """
    McConnaughey distance for vectors of binary values.

    Computes the McConnaughey distance [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices by subtracting the similarity from 1,
    using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`mcconnaughey_binary_similarity`. It uses the normalized
    similarity, scaled to range `[0, 1]`.
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
        McConnaughey distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `McConnaughey B.H.
        "The determination and analysis of plankton communities"
        Lembaga Penelitian Laut, 1964.
        <https://books.google.pl/books?id=7aBbOQAACAAJ>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import mcconnaughey_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = mcconnaughey_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = mcconnaughey_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - mcconnaughey_binary_similarity(vec_a, vec_b, normalized=True)


@validate_params(
    {"X": ["array-like"], "Y": ["array-like", None]},
    prefer_skip_nested_validation=True,
)
def bulk_mcconnaughey_binary_similarity(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    normalized: bool = False,
) -> np.ndarray:
    r"""
    Bulk McConnaughey similarity for binary matrices.

    Computes the pairwise McConnaughey similarity between binary matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`mcconnaughey_binary_similarity`.

    Parameters
    ----------
    X : ndarray
        First binary input array, of shape :math:`m \times m`

    Y : ndarray, default=None
        Second binary input array, of shape :math:`n \times n`. If not passed, similarities
        are computed between rows of X.

    normalized : bool, default=False
        Whether to normalize the values inside the result matrix to range ``[0, 1]`` by adding
        one and dividing the result by 2.

    Returns
    -------
    similarities : ndarray
        Array with pairwise McConnaughey similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`mcconnaughey_binary_similarity` : McConnaughey similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_mcconnaughey_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 1, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_mcconnaughey_binary_similarity(X, Y)
    >>> sim
    array([[0.66666667, 0.5       ],
           [0.5       , 0.5       ]])
    """
    if Y is None:
        return _bulk_mcconnaughey_binary_similarity_single(X, normalized)
    else:
        return _bulk_mcconnaughey_binary_similarity_two(X, Y, normalized)


@numba.njit(parallel=True)
def _bulk_mcconnaughey_binary_similarity_single(
    X: np.ndarray,
    normalized: bool,
) -> np.ndarray:
    m = X.shape[0]
    sims = np.empty((m, m))
    X_sum = np.sum(X, axis=1)

    # upper triangle - actual similarities
    for i in numba.prange(m):
        vec_a = X[i]
        sum_a = X_sum[i]

        for j in numba.prange(i, m):
            vec_b = X[j]
            sum_b = X_sum[j]

            num_common = np.sum(np.logical_and(vec_a, vec_b))
            sum_ab = sum_a + sum_b
            dot_ab = sum_a * sum_b

            if sum_ab == 0:
                sim = 1.0
            elif dot_ab == 0:
                sim = -1.0
            else:
                sim = (num_common * sum_ab - dot_ab) / dot_ab
                if normalized:
                    sim = (sim + 1) / 2

            sims[i, j] = sims[j, i] = sim

    return sims


@numba.njit(parallel=True)
def _bulk_mcconnaughey_binary_similarity_two(
    X: np.ndarray,
    Y: np.ndarray,
    normalized: bool,
) -> np.ndarray:
    m = X.shape[0]
    n = Y.shape[0]
    sims = np.empty((m, n))
    X_sum = np.sum(X, axis=1)
    Y_sum = np.sum(Y, axis=1)

    for i in numba.prange(m):
        vec_a = X[i]
        sum_a = X_sum[i]

        for j in numba.prange(n):
            vec_b = Y[j]
            sum_b = Y_sum[j]

            num_common = np.sum(np.logical_and(vec_a, vec_b))
            sum_ab = sum_a + sum_b
            dot_ab = sum_a * sum_b

            if sum_ab == 0:
                sim = 1.0
            elif dot_ab == 0:
                sim = -1.0
            else:
                sim = (num_common * sum_ab - dot_ab) / dot_ab
                if normalized:
                    sim = (sim + 1) / 2

            sims[i, j] = sims[j, i] = sim

    return sims


@validate_params(
    {
        "X": ["array-like", csr_array],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_mcconnaughey_binary_distance(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    Bulk McConnaughey distance for vectors of binary values.

    Computes the pairwise McConnaughey distance between binary matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`mcconnaughey_binary_distance`.

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
        Array with pairwise McConnaughey distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`mcconnaughey_binary_distance` : McConnaughey distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_mcconnaughey_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 1, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 1, 0]])
    >>> dist = bulk_mcconnaughey_binary_distance(X, Y)
    >>> dist
    array([[0.16666667, 0.        ],
           [0.        , 0.5       ]])

    >>> X = np.array([[1, 1, 1], [1, 0, 0]])
    >>> dist = bulk_mcconnaughey_binary_distance(X)
    >>> dist
    array([[0.        , 0.33333333],
           [0.33333333, 0.        ]])
    """
    return 1 - bulk_mcconnaughey_binary_similarity(X, Y, normalized=True)
