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
def rand_binary_similarity(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    r"""
    Rand similarity for vectors of binary values.

    Computes the Rand similarity [1]_ [2]_ (known as All-Bit [3]_ or Sokal-Michener)
    for binary data between two input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{|a \cap b|}{n}

    where `n` is the length of vector `a`.

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
        Rand similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Rand, W.M.
        "Objective criteria for the evaluation of clustering methods."
        J. Amer. Stat. Assoc. 1971; 66: 846–850.
        <https://www.tandfonline.com/doi/abs/10.1080/01621459.1971.10482356>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import rand_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = rand_binary_similarity(vec_a, vec_b)
    >>> sim
    0.6666666666666666

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = rand_binary_similarity(vec_a, vec_b)
    >>> sim
    0.6666666666666666
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if isinstance(vec_a, (np.ndarray, list)):
        num_common = np.sum(np.logical_and(vec_a, vec_b))
        length = len(vec_a)
    else:
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)
        num_common = len(vec_a_idxs & vec_b_idxs)
        length = vec_a.shape[1]

    rand_sim = num_common / length
    return float(rand_sim)


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def rand_binary_distance(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    """
    Rand distance for vectors of binary values.

    Computes the Rand distance [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices by subtracting the similarity from 1,
    using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`rand_binary_similarity`.
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
        Rand distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Rand, W.M.
        "Objective criteria for the evaluation of clustering methods."
        J. Amer. Stat. Assoc. 1971; 66: 846–850.
        <https://www.tandfonline.com/doi/abs/10.1080/01621459.1971.10482356>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import rand_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = rand_binary_distance(vec_a, vec_b)
    >>> dist
    0.33333333333333337

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = rand_binary_distance(vec_a, vec_b)
    >>> dist
    0.33333333333333337
    """
    return 1 - rand_binary_similarity(vec_a, vec_b)


@validate_params(
    {"X": ["array-like"], "Y": ["array-like", None]},
    prefer_skip_nested_validation=True,
)
def bulk_rand_binary_similarity(
    X: np.ndarray, Y: np.ndarray | None = None
) -> np.ndarray:
    r"""
    Bulk Rand similarity for binary matrices.

    Computes the pairwise Rand [1]_ [2]_ (known as All-Bit [3]_ or Sokal-Michener)
    similarity between binary matrices. If one array is passed, similarities are
    computed between its rows. For two arrays, similarities are between their respective
    rows, with `i`-th row and `j`-th column in output corresponding to `i`-th row from
    first array and `j`-th row from second array.

    See also :py:func:`rand_binary_similarity`.

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
        Array with pairwise Rand similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    References
    ----------
    .. [1] `Rand, W.M.
        "Objective criteria for the evaluation of clustering methods."
        J. Amer. Stat. Assoc. 1971; 66: 846–850.
        <https://www.tandfonline.com/doi/abs/10.1080/01621459.1971.10482356>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    See Also
    --------
    :py:func:`rand_binary_similarity` : Rand similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_rand_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_rand_binary_similarity(X, Y)
    >>> sim
    array([[0.66666667, 0.33333333],
           [0.33333333, 0.33333333]])
    """
    if Y is None:
        return _bulk_rand_binary_similarity_single(X)
    else:
        return _bulk_rand_binary_similarity_two(X, Y)


@numba.njit(parallel=True)
def _bulk_rand_binary_similarity_single(X: np.ndarray) -> np.ndarray:
    m, length = X.shape
    sims = np.empty((m, m))

    for i in numba.prange(m):
        for j in numba.prange(i, m):
            intersection = np.sum(np.logical_and(X[i], X[j]))
            sim = intersection / length
            sims[i, j] = sims[j, i] = sim

    return sims


@numba.njit(parallel=True)
def _bulk_rand_binary_similarity_two(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    m, length = X.shape
    n = Y.shape[0]
    sims = np.empty((m, n))

    for i in numba.prange(m):
        for j in numba.prange(n):
            intersection = np.sum(np.logical_and(X[i], Y[j]))
            sims[i, j] = intersection / length

    return sims


@validate_params(
    {
        "X": ["array-like", csr_array],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_rand_binary_distance(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    r"""
    Bulk Rand distance for vectors of binary values.

    Computes the pairwise Rand distance between binary matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`rand_binary_distance`.

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
        Array with pairwise Rand distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`rand_binary_distance` : Rand distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_rand_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_rand_binary_distance(X, Y)
    >>> dist
    array([[0.33333333, 0.33333333],
           [0.33333333, 0.33333333]])

    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_rand_binary_distance(X)
    >>> dist
    array([[0.33333333, 0.33333333],
           [0.33333333, 0.33333333]])
    """
    return 1 - bulk_rand_binary_similarity(X, Y)
