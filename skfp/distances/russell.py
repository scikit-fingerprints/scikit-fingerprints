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
def russell_binary_similarity(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    r"""
    Russell similarity for vectors of binary values.

    Computes the Russell similarity [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices, using the formula:

    .. math::

        sim(x, y) = \frac{a}{n}

    where

    - :math:`a` - common "on" bits
    - :math:`n` - length of passed vectors

    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a similarity of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        Russell similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Russell P.F., Rao T.R.
        "On habitat and association of species of anopheline larvae in south-eastern Madras"
        Journal of the Malaria Institute of India, 1940, June, Vol. 3, No. 1, 153-178 pp.
        <https://www.cabidigitallibrary.org/doi/full/10.5555/19412900343>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import russell_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 1, 1, 1])
    >>> vec_b = np.array([1, 1, 0, 0])
    >>> sim = russell_binary_similarity(vec_a, vec_b)
    >>> sim
    0.5

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 1, 1, 1]])
    >>> vec_b = csr_array([[1, 1, 0, 0]])
    >>> sim = russell_binary_similarity(vec_a, vec_b)
    >>> sim
    0.5
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if isinstance(vec_a, np.ndarray):
        a = np.sum(np.logical_and(vec_a, vec_b))
        n = len(vec_a)
    else:
        n = vec_a.shape[1]
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)

        a = len(vec_a_idxs & vec_b_idxs)

    sim = a / n

    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def russell_binary_distance(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    """
    Russell distance for vectors of binary values.

    Computes the Russell distance [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices by subtracting the similarity from 1,
    using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`russell_binary_similarity`.
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
    distance : float
        Russell distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Russell P.F., Rao T.R.
        "On habitat and association of species of anopheline larvae in south-eastern Madras"
        Journal of the Malaria Institute of India, 1940, June, Vol. 3, No. 1, 153-178 pp.
        <https://www.cabidigitallibrary.org/doi/full/10.5555/19412900343>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import russell_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 1, 1, 1])
    >>> vec_b = np.array([1, 1, 0, 0])
    >>> dist = russell_binary_distance(vec_a, vec_b)
    >>> dist
    0.5

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 1, 1, 1]])
    >>> vec_b = csr_array([[1, 1, 0, 0]])
    >>> dist = russell_binary_distance(vec_a, vec_b)
    >>> dist
    0.5
    """
    return 1 - russell_binary_similarity(vec_a, vec_b)


@validate_params(
    {"X": ["array-like"], "Y": ["array-like", None]},
    prefer_skip_nested_validation=True,
)
def bulk_russell_binary_similarity(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""
    Bulk Russell similarity for binary matrices.

    Computes the pairwise Russell similarity between binary matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`russell_binary_similarity`.

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
        Array with pairwise Russell similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`russell_binary_similarity` : Russell similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_russell_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 1, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_russell_binary_similarity(X, Y)
    >>> sim
    array([[0.66666667, 0.66666667],
           [0.33333333, 0.33333333]])
    """
    if Y is None:
        return _bulk_russell_binary_similarity_single(X)
    else:
        return _bulk_russell_binary_similarity_two(X, Y)


@numba.njit(parallel=True)
def _bulk_russell_binary_similarity_single(
    X: np.ndarray,
) -> np.ndarray:
    m = X.shape[0]
    sims = np.empty((m, m))

    # upper triangle - actual similarities
    for i in numba.prange(m):
        vec_a = X[i]
        len_a = len(vec_a)

        for j in numba.prange(i, m):
            vec_b = X[j]

            a = np.sum(np.logical_and(vec_a, vec_b))

            sim = a / len_a
            sims[i, j] = sim

    # lower triangle - symmetric with upper triangle
    for i in numba.prange(1, m):
        for j in numba.prange(i):
            sims[i, j] = sims[j, i]

    return sims


@numba.njit(parallel=True)
def _bulk_russell_binary_similarity_two(
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    m = X.shape[0]
    n = Y.shape[0]
    sims = np.empty((m, n))

    for i in numba.prange(m):
        vec_a = X[i]
        len_a = len(vec_a)

        for j in numba.prange(n):
            vec_b = Y[j]

            a = np.sum(np.logical_and(vec_a, vec_b))

            sim = a / len_a
            sims[i, j] = sim

    return sims


@validate_params(
    {
        "X": ["array-like", csr_array],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_russell_binary_distance(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    Bulk Russell distance for vectors of binary values.

    Computes the pairwise Russell distance between binary matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`russell_binary_distance`.

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
        Array with pairwise Russell distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`russell_binary_distance` : Russell distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_russell_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 1, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 1, 0]])
    >>> dist = bulk_russell_binary_distance(X, Y)
    >>> dist
    array([[0.33333333, 0.33333333],
           [0.33333333, 0.66666667]])

    >>> X = np.array([[1, 1, 1], [1, 0, 0]])
    >>> dist = bulk_russell_binary_distance(X)
    >>> dist
    array([[0.        , 0.66666667],
           [0.66666667, 0.66666667]])
    """
    return 1 - bulk_russell_binary_similarity(X, Y)
