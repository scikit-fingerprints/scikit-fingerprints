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
def sokal_sneath_2_binary_similarity(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    r"""
    Sokal-Sneath similarity 2 for vectors of binary values.

    Computes the Sokal-Sneath similarity 2 [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{|a \cap b|}{|a \cup b| + |a \Delta b} =
                    \frac{|a \cap b|}{2 * |a| + 2 * |b| - 3 * |a \cap b|}

    where :`|a \Delta b|` is the XOR operation (symmetric difference), i.e. number
    of bits that are "on" in one vector and "off" in another.

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
        Sokal-Sneath similarity 2 between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `R. R. Sokal, P. H. A. Sneath
        "Principles of Numerical Taxonomy"
        Principles of Numerical Taxonomy., 1963, 359 ref. bibl. 18 pp.
        <https://www.cabidigitallibrary.org/doi/full/10.5555/19650300280>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import sokal_sneath_2_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 1, 1, 1])
    >>> vec_b = np.array([1, 1, 0, 0])
    >>> sim = sokal_sneath_2_binary_similarity(vec_a, vec_b)
    >>> sim
    0.3333333333333333

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 1, 1, 1]])
    >>> vec_b = csr_array([[1, 1, 0, 0]])
    >>> sim = sokal_sneath_2_binary_similarity(vec_a, vec_b)
    >>> sim
    0.3333333333333333
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if isinstance(vec_a, (np.ndarray, list)):
        intersection = np.sum(np.logical_and(vec_a, vec_b))
        a_sum = np.sum(vec_a)
        b_sum = np.sum(vec_b)
    else:
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)

        intersection = len(vec_a_idxs & vec_b_idxs)
        a_sum = len(vec_a_idxs)
        b_sum = len(vec_b_idxs)

    denominator = 2 * a_sum + 2 * b_sum - 3 * intersection
    sim = intersection / denominator if denominator > 0 else 1.0

    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def sokal_sneath_2_binary_distance(
    vec_a: list | np.ndarray | csr_array,
    vec_b: list | np.ndarray | csr_array,
) -> float:
    """
    Sokal-Sneath distance 2 for vectors of binary values.

    Computes the Sokal-Sneath distance 2 [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices by subtracting the similarity from 1,
    using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`sokal_sneath_2_binary_similarity`.
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
        Sokal-Sneath distance 2 between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `R. R. Sokal, P. H. A. Sneath
        "Principles of Numerical Taxonomy"
        Principles of Numerical Taxonomy., 1963, 359 ref. bibl. 18 pp.
        <https://www.cabidigitallibrary.org/doi/full/10.5555/19650300280>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import sokal_sneath_2_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 1, 1, 1])
    >>> vec_b = np.array([1, 1, 0, 0])
    >>> dist = sokal_sneath_2_binary_distance(vec_a, vec_b)
    >>> dist
    0.6666666666666667

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 1, 1, 1]])
    >>> vec_b = csr_array([[1, 1, 0, 0]])
    >>> dist = sokal_sneath_2_binary_distance(vec_a, vec_b)
    >>> dist
    0.6666666666666667
    """
    return 1 - sokal_sneath_2_binary_similarity(vec_a, vec_b)


@validate_params(
    {"X": ["array-like"], "Y": ["array-like", None]},
    prefer_skip_nested_validation=True,
)
def bulk_sokal_sneath_2_binary_similarity(
    X: np.ndarray,
    Y: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    Bulk Sokal-Sneath similarity 2 for binary matrices.

    Computes the pairwise Sokal-Sneath similarity 2 between binary matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`sokal_sneath_2_binary_similarity`.

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
        Array with pairwise Sokal-Sneath similarity 2 values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`sokal_sneath_2_binary_similarity` : Sokal-Sneath similarity 2 function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_sokal_sneath_2_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 1, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_sokal_sneath_2_binary_similarity(X, Y)
    >>> sim
    array([[0.5       , 0.5       ],
           [0.33333333, 0.33333333]])
    """
    if Y is None:
        return _bulk_sokal_sneath_2_binary_similarity_single(X)
    else:
        return _bulk_sokal_sneath_2_binary_similarity_two(X, Y)


@numba.njit(parallel=True)
def _bulk_sokal_sneath_2_binary_similarity_single(
    X: np.ndarray,
) -> np.ndarray:
    m = X.shape[0]
    sims = np.empty((m, m))
    X_sum = np.sum(X, axis=1)

    # upper triangle - actual similarities
    for i in numba.prange(m):
        vec_a = X[i]
        sum_a = X_sum[i]
        sims[i, i] = 1.0

        for j in numba.prange(i + 1, m):
            vec_b = X[j]
            sum_b = X_sum[j]

            intersection = np.sum(np.logical_and(vec_a, vec_b))

            denominator = 2 * sum_a + 2 * sum_b - 3 * intersection
            sim = intersection / denominator if denominator > 0 else 1.0

            sims[i, j] = sims[j, i] = sim

    return sims


@numba.njit(parallel=True)
def _bulk_sokal_sneath_2_binary_similarity_two(
    X: np.ndarray,
    Y: np.ndarray,
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

            intersection = np.sum(np.logical_and(vec_a, vec_b))

            denominator = 2 * sum_a + 2 * sum_b - 3 * intersection
            sim = intersection / denominator if denominator > 0 else 1.0

            sims[i, j] = sim

    return sims


@validate_params(
    {
        "X": ["array-like", csr_array],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_sokal_sneath_2_binary_distance(
    X: np.ndarray, Y: np.ndarray | None = None
) -> np.ndarray:
    r"""
    Bulk Sokal-Sneath distance 2 for vectors of binary values.

    Computes the pairwise Sokal-Sneath distance 2 between binary matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`sokal_sneath_2_binary_distance`.

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
        Array with pairwise Sokal-Sneath distance 2 values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`sokal_sneath_2_binary_distance` : Sokal-Sneath distance 2 function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_sokal_sneath_2_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 1, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 1, 0]])
    >>> dist = bulk_sokal_sneath_2_binary_distance(X, Y)
    >>> dist
    array([[0.5, 0.5],
           [0. , 0.8]])

    >>> X = np.array([[1, 1, 1], [1, 0, 0]])
    >>> dist = bulk_sokal_sneath_2_binary_distance(X)
    >>> dist
    array([[0. , 0.8],
           [0.8, 0. ]])
    """
    return 1 - bulk_sokal_sneath_2_binary_similarity(X, Y)
