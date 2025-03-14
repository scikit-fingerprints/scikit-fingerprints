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
def harris_lahey_binary_similarity(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
    normalized: bool = False,
) -> float:
    r"""
    Harris-Lahey similarity for vectors of binary values.

    Computes the Harris-Lahey similarity [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices, using the formula:

    .. math::

        sim(x, y) = \frac{a * (2d + b + c)}{2 * (a + b + c)} +
                    \frac{d * (2a + b + c)}{2 * (b + c + d)}

    where :math:`a`, :math:`b`, :math:`c` and :math:`d` correspond to the number
    of bit relations between the two vectors:

    - :math:`a` - both are 1 (:math:`|x \cap y|`, common "on" bits)
    - :math:`b` - :math:`x` is 1, :math:`y` is 0
    - :math:`c` - :math:`x` is 0, :math:`y` is 1
    - :math:`d` - both are 0

    The calculated similarity falls within the range :math:`[0, n]`, where :math:`n`
    is the length of vectors. Use ``normalized`` argument to scale the similarity to
    range :math:`[0, 1]`.
    Passing all-zero or all-one vectors to this function results in a similarity of
    :math:`n`.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    normalized : bool, default=False
        Whether to divide the resulting similarity by length of vectors (their number
        of elements), to normalize values to range ``[0, 1]``.

    Returns
    -------
    similarity : float
        Harris-Lahey similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Francis C. Harris, Benjamin B. Lahey
        "A method for combining occurrence and nonoccurrence interobserver agreement scores"
        J Appl Behav Anal. 1978 Winter;11(4):523-7.
        <https://doi.org/10.1901/jaba.1978.11-523>`_

    .. [2] `Brusco M, Cradit JD, Steinley D.
        "A comparison of 71 binary similarity coefficients: The effect of base rates"
        PLoS One. 2021 Apr 7;16(4):e0247751.
        <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0247751>`_

    .. [3] `R. Todeschini, V. Consonni, H. Xiang, J. Holliday, M. Buscema, P. Willett
        "Similarity Coefficients for Binary Chemoinformatics Data: Overview and
        Extended Comparison Using Simulated and Real Data Sets"
        J. Chem. Inf. Model. 2012, 52, 11, 2884–2901
        <https://doi.org/10.1021/ci300261r>`_

    Examples
    --------
    >>> from skfp.distances import harris_lahey_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = harris_lahey_binary_similarity(vec_a, vec_b)
    >>> sim
    3.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = harris_lahey_binary_similarity(vec_a, vec_b)
    >>> sim
    3.0
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if isinstance(vec_a, np.ndarray):
        length = len(vec_a)
        vec_a_neg = 1 - vec_a
        vec_b_neg = 1 - vec_b

        a = np.sum(np.logical_and(vec_a, vec_b))
        b = np.sum(np.logical_and(vec_a, vec_b_neg))
        c = np.sum(np.logical_and(vec_a_neg, vec_b))
        d = np.sum(np.logical_and(vec_a_neg, vec_b_neg))
    else:
        length = vec_a.shape[1]
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)

        a = len(vec_a_idxs & vec_b_idxs)
        b = len(vec_a_idxs - vec_b_idxs)
        c = len(vec_b_idxs - vec_a_idxs)
        d = length - (a + b + c)

    first_denom = a + b + c
    second_denom = b + c + d

    # all-ones or all-zeros vectors
    if first_denom == 0 or second_denom == 0:
        return 1.0

    sim = float(
        (a * (2 * d + b + c)) / (2 * first_denom)
        + (d * (2 * a + b + c)) / (2 * second_denom)
    )

    if normalized:
        sim = sim / length

    return sim


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def harris_lahey_binary_distance(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    """
    Harris-Lahey distance for vectors of binary values.

    Computes the Harris-Lahey distance [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices by subtracting the similarity from 1,
    using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`harris_lahey_binary_similarity`. It uses the normalized
    similarity, scaled to range `[0, 1]`.
    The calculated distance falls within the range :math:`[0, 1]`.
    Passing all-zero or all-ones vectors to this function results in a distance of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    distance : float
        Harris-Lahey distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Francis C. Harris, Benjamin B. Lahey
        "A method for combining occurrence and nonoccurrence interobserver agreement scores"
        J Appl Behav Anal. 1978 Winter;11(4):523-7.
        <https://doi.org/10.1901/jaba.1978.11-523>`_

    .. [2] `Brusco M, Cradit JD, Steinley D.
        "A comparison of 71 binary similarity coefficients: The effect of base rates"
        PLoS One. 2021 Apr 7;16(4):e0247751.
        <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0247751>`_

    .. [3] `R. Todeschini, V. Consonni, H. Xiang, J. Holliday, M. Buscema, P. Willett
        "Similarity Coefficients for Binary Chemoinformatics Data: Overview and
        Extended Comparison Using Simulated and Real Data Sets"
        J. Chem. Inf. Model. 2012, 52, 11, 2884–2901
        <https://doi.org/10.1021/ci300261r>`_

    Examples
    --------
    >>> from skfp.distances import harris_lahey_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = harris_lahey_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = harris_lahey_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - harris_lahey_binary_similarity(vec_a, vec_b, normalized=True)


@validate_params(
    {"X": ["array-like"], "Y": ["array-like", None]},
    prefer_skip_nested_validation=True,
)
def bulk_harris_lahey_binary_similarity(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    normalized: bool = False,
) -> np.ndarray:
    r"""
    Bulk Harris-Lahey similarity for binary matrices.

    Computes the pairwise Harris-Lahey similarity between binary matrices. If one array is
    passed, similarities are computed between its rows. For two arrays, similarities
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`harris_lahey_binary_similarity`.

    Parameters
    ----------
    X : ndarray
        First binary input array, of shape :math:`m \times m`

    Y : ndarray, default=None
        Second binary input array, of shape :math:`n \times n`. If not passed, similarities
        are computed between rows of X.

    normalized : bool, default=False
        Whether to divide the resulting similarity by length of vectors, (their number
        of elements), to normalize values to range ``[0, 1]``.

    Returns
    -------
    similarities : ndarray
        Array with pairwise Harris-Lahey similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`harris_lahey_binary_similarity` : Harris-Lahey similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_harris_lahey_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_harris_lahey_binary_similarity(X, Y)
    >>> sim
    array([[3.        , 0.33333333],
           [1.5       , 1.5       ]])
    """
    if Y is None:
        return _bulk_harris_lahey_binary_similarity_single(X, normalized)
    else:
        return _bulk_harris_lahey_binary_similarity_two(X, Y, normalized)


@numba.njit(parallel=True)
def _bulk_harris_lahey_binary_similarity_single(
    X: np.ndarray, normalized: bool
) -> np.ndarray:
    m, length = X.shape
    sims = np.empty((m, m))

    for i in numba.prange(m):
        vec_a = X[i]
        vec_a_neg = 1 - vec_a

        for j in numba.prange(i, m):
            vec_b = X[j]
            vec_b_neg = 1 - vec_b

            a = np.sum(np.logical_and(vec_a, vec_b))
            b = np.sum(np.logical_and(vec_a, vec_b_neg))
            c = np.sum(np.logical_and(vec_a_neg, vec_b))
            d = np.sum(np.logical_and(vec_a_neg, vec_b_neg))

            bc_sum = b + c

            first_denom = a + bc_sum
            second_denom = bc_sum + d

            # all-ones or all-zeros vectors
            if first_denom == 0 or second_denom == 0:
                sim = 1.0
            else:
                sim = float(
                    (a * (2 * d + bc_sum)) / (2 * first_denom)
                    + (d * (2 * a + bc_sum)) / (2 * second_denom)
                )
                if normalized:
                    sim /= length

            sims[i, j] = sims[j, i] = sim

    return sims


@numba.njit(parallel=True)
def _bulk_harris_lahey_binary_similarity_two(
    X: np.ndarray, Y: np.ndarray, normalized: bool
) -> np.ndarray:
    m, length = X.shape
    n = Y.shape[0]
    sims = np.empty((m, n))

    for i in numba.prange(m):
        vec_a = X[i]
        vec_a_neg = 1 - vec_a

        for j in numba.prange(n):
            vec_b = Y[j]
            vec_b_neg = 1 - vec_b

            a = np.sum(np.logical_and(vec_a, vec_b))
            b = np.sum(np.logical_and(vec_a, vec_b_neg))
            c = np.sum(np.logical_and(vec_a_neg, vec_b))
            d = np.sum(np.logical_and(vec_a_neg, vec_b_neg))

            bc_sum = b + c

            first_denom = a + bc_sum
            second_denom = bc_sum + d

            # all-ones or all-zeros vectors
            if first_denom == 0 or second_denom == 0:
                sim = 1.0
            else:
                sim = float(
                    (a * (2 * d + bc_sum)) / (2 * first_denom)
                    + (d * (2 * a + bc_sum)) / (2 * second_denom)
                )
                if normalized:
                    sim /= length

            sims[i, j] = sim

    return sims


@validate_params(
    {
        "X": ["array-like", csr_array],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_harris_lahey_binary_distance(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    Bulk Harris-Lahey distance for vectors of binary values.

    Computes the pairwise Harris-Lahey distance between binary matrices. If one array is
    passed, distances are computed between its rows. For two arrays, distances
    are between their respective rows, with `i`-th row and `j`-th column in output
    corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`harris_lahey_binary_distance`.

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
        Array with pairwise Harris-Lahey distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`harris_lahey_binary_distance` : Harris-Lahey distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_harris_lahey_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_harris_lahey_binary_distance(X, Y)
    >>> dist
    array([[0., 0.],
           [0., 0.]])

    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_harris_lahey_binary_distance(X)
    >>> dist
    array([[0., 0.],
           [0., 0.]])
    """
    return 1 - bulk_harris_lahey_binary_similarity(X, Y, normalized=True)
