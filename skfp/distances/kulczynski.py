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
def kulczynski_binary_similarity(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    r"""
    Kulczynski similarity for vectors of binary values.

    Computes the Kulczynski II similarity [1]_ [2]_ [3]_ for binary data between
    two input arrays or sparse matrices using the formula:

    .. math::

        sim(x, y) = \frac{1}{2} \left( \frac{a}{a+b} + \frac{a}{a+c} \right)

    where :math:`a`, :math:`b` and :math:`c` correspond to the number of bit
    relations between the two vectors:

    - :math:`a` - both are 1 (:math:`|x \cap y|`, common "on" bits)
    - :math:`b` - :math:`x` is 1, :math:`y` is 0
    - :math:`c` - :math:`x` is 0, :math:`y` is 1

    Note that this is the second Kulczynski similarity, also used by RDKit. It
    differs from Kulczynski I similarity from e.g. SciPy.

    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing two all-zero vectors to this function results in a similarity of 1.
    However, when only one is all-zero (i.e. :math:`a+b=0` or :math:`a+c=0`), the
    similarity is 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        Kulczynski similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] Kulczynski, S.
        "Zespoly roslin w Pieninach."
        Bull Int l’Academie Pol des Sci des Lettres (1927): 57-203.

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import kulczynski_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = kulczynski_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = kulczynski_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if np.sum(vec_a) == 0 == np.sum(vec_b):
        return 1.0

    if isinstance(vec_a, np.ndarray):
        a = np.sum(np.logical_and(vec_a, vec_b))
        b = np.sum(np.logical_and(vec_a, 1 - vec_b))
        c = np.sum(np.logical_and(1 - vec_a, vec_b))
    else:
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)

        a = len(vec_a_idxs & vec_b_idxs)
        b = len(vec_a_idxs - vec_b_idxs)
        c = len(vec_b_idxs - vec_a_idxs)

    if a + b == 0 or a + c == 0:
        return 0.0

    sim = (a / (a + b) + a / (a + c)) / 2
    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def kulczynski_binary_distance(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    """
    Kulczynski distance for vectors of binary values.

    Computes the Kulczynski II distance for binary data between two input arrays
    or sparse matrices by subtracting the Kulczynski II similarity [1]_ [2]_ [3]_
    from 1, using to the formula:

    .. math::

        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`kulczynski_binary_similarity`.
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
        Kulczynski distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] Kulczynski, S.
        "Zespoly roslin w Pieninach."
        Bull Int l’Academie Pol des Sci des Lettres (1927): 57-203.

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import kulczynski_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = kulczynski_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = kulczynski_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - kulczynski_binary_similarity(vec_a, vec_b)


@validate_params(
    {"X": ["array-like"], "Y": ["array-like", None]},
    prefer_skip_nested_validation=True,
)
def bulk_kulczynski_binary_similarity(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""
    Bulk Kulczynski similarity for binary matrices.

    Computes the pairwise Kulczynski similarity between binary matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`kulczynski_binary_similarity`.

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
        Array with pairwise Kulczynski similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`kulczynski_binary_similarity` : Kulczynski similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_kulczynski_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_kulczynski_binary_similarity(X, Y)
    >>> sim
    array([[1.  , 0.5 ],
           [0.75, 0.75]])
    """
    if Y is None:
        return _bulk_kulczynski_binary_similarity_single(X)
    else:
        return _bulk_kulczynski_binary_similarity_two(X, Y)


@numba.njit(parallel=True)
def _bulk_kulczynski_binary_similarity_single(
    X: np.ndarray,
) -> np.ndarray:
    m = X.shape[0]
    sims = np.empty((m, m))
    sum_X = np.sum(X, axis=1)

    # upper triangle - actual similarities
    for i in numba.prange(m):
        vec_a = X[i]
        sum_a = sum_X[i]
        vec_a_neg = 1 - vec_a

        for j in numba.prange(i, m):
            vec_b = X[j]
            sum_b = sum_X[j]

            if sum_a == 0 == sum_b:
                sims[i, j] = 1.0
                continue

            # no need to compute vec_b_neg if sum_a == 0 == sum_b
            vec_b_neg = 1 - vec_b

            a = np.sum(np.logical_and(vec_a, vec_b))
            b = np.sum(np.logical_and(vec_a, vec_b_neg))
            c = np.sum(np.logical_and(vec_a_neg, vec_b))

            if a + b == 0 or a + c == 0:
                sims[i, j] = 0.0
                continue

            sims[i, j] = (a / (a + b) + a / (a + c)) / 2.0

    # lower triangle - symmetric with upper triangle
    for i in numba.prange(1, m):
        for j in numba.prange(i):
            sims[i, j] = sims[j, i]

    return sims


@numba.njit(parallel=True)
def _bulk_kulczynski_binary_similarity_two(
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    m = X.shape[0]
    n = Y.shape[0]
    sims = np.empty((m, n))
    sum_X = np.sum(X, axis=1)
    sum_Y = np.sum(Y, axis=1)

    for i in numba.prange(m):
        vec_a = X[i]
        sum_a = sum_X[i]
        vec_a_neg = 1 - vec_a

        for j in numba.prange(n):
            vec_b = Y[j]
            sum_b = sum_Y[j]

            if sum_a == 0 == sum_b:
                sims[i, j] = 1.0
                continue

            # no need to compute vec_b_neg if sum_a == 0 == sum_b
            vec_b_neg = 1 - vec_b

            a = np.sum(np.logical_and(vec_a, vec_b))
            b = np.sum(np.logical_and(vec_a, vec_b_neg))
            c = np.sum(np.logical_and(vec_a_neg, vec_b))

            if a + b == 0 or a + c == 0:
                sims[i, j] = 0.0
                continue

            sims[i, j] = (a / (a + b) + a / (a + c)) / 2.0

    return sims


@validate_params(
    {
        "X": ["array-like", csr_array],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_kulczynski_binary_distance(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    Bulk Kulczynski distance for vectors of binary values.

    Computes the pairwise Kulczynski distance between binary matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`kulczynski_binary_distance`.

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
        Array with pairwise Kulczynski distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`kulczynski_binary_distance` : Kulczynski distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_kulczynski_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_kulczynski_binary_distance(X, Y)
    >>> dist
    array([[0., 0.],
           [0., 0.]])

    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_kulczynski_binary_distance(X)
    >>> dist
    array([[0., 0.],
           [0., 0.]])
    """
    return 1 - bulk_kulczynski_binary_similarity(X, Y)
