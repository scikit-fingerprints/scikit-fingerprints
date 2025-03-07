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
def ct4_binary_similarity(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    r"""
    Consonni–Todeschini 4 similarity for vectors of binary values.

    Computes the Consonni–Todeschini 4 (CT4) similarity [1]_ [2]_ [3]_ for binary data
    between two input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{\log (1 + |a \cap b|)}{\log (1 + |a \cup b|)}
        = \frac{\log (1 + |a \cap b|)}{\log (1 + |a| + |b| - |a \cap b|)}

    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in similarity of 1.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        CT4 similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `V. Consonni, R. Todeschini
        "New similarity coefficients for binary data"
        MATCH Commun.Math.Comput.Chem.. 68. 581-592.
        <https://match.pmf.kg.ac.rs/electronic_versions/Match68/n2/match68n2_581-592.pdf>`_

    .. [2] `Todeschini, Roberto, Davide Ballabio, and Viviana Consonni
        "Distances and similarity measures in chemometrics and chemoinformatics."
        Encyclopedia of Analytical Chemistry: Applications, Theory and Instrumentation (2006): 1-40.
        <https://doi.org/10.1002/9780470027318.a9438.pub2>`_

    .. [3] `Todeschini, Roberto, et al.
        "Similarity coefficients for binary chemoinformatics data: overview and
        extended comparison using simulated and real data sets."
        J. Chem. Inf. Model. 2012, 52, 11, 2884–2901
        <https://doi.org/10.1021/ci300261r>`_

    Examples
    --------
    >>> from skfp.distances import ct4_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = ct4_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = ct4_binary_similarity(vec_a, vec_b)
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

    sim = float(np.log(1 + intersection) / np.log(1 + union)) if union != 0 else 1.0
    return sim


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def ct4_binary_distance(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    """
    Consonni–Todeschini 4 distance for vectors of binary values.

    Computes the Consonni–Todeschini 4 (CT4) distance [1]_ [2]_ [3]_ for binary data
    between two input arrays or sparse matrices by subtracting the similarity
    from 1, using the formula:

    .. math::

        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`ct4_binary_similarity`.
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
    .. [1] `V. Consonni, R. Todeschini
        "New similarity coefficients for binary data"
        MATCH Commun.Math.Comput.Chem.. 68. 581-592.
        <https://match.pmf.kg.ac.rs/electronic_versions/Match68/n2/match68n2_581-592.pdf>`_

    .. [2] `Todeschini, Roberto, Davide Ballabio, and Viviana Consonni
        "Distances and similarity measures in chemometrics and chemoinformatics."
        Encyclopedia of Analytical Chemistry: Applications, Theory and Instrumentation (2006): 1-40.
        <https://doi.org/10.1002/9780470027318.a9438.pub2>`_

    .. [3] `Todeschini, Roberto, et al.
        "Similarity coefficients for binary chemoinformatics data: overview and
        extended comparison using simulated and real data sets."
        J. Chem. Inf. Model. 2012, 52, 11, 2884–2901
        <https://doi.org/10.1021/ci300261r>`_

    Returns
    -------
    distance : float
        CT4 distance between ``vec_a`` and ``vec_b``.

    Examples
    --------
    >>> from skfp.distances import ct4_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = ct4_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = ct4_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - ct4_binary_similarity(vec_a, vec_b)


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def ct4_count_similarity(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    r"""
    Consonni–Todeschini 4 similarity for vectors of count values.

    Computes the Consonni–Todeschini 4 (CT4) similarity [1]_ [2]_ [3]_ for count data
    between two input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{\log (1 + a \cdot b)}{\log (1 + \|a\|^2 + \|b\|^2 - a \cdot b)}

    The calculated similarity falls within the range :math:`[0, 1]`.
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
        CT4 similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `V. Consonni, R. Todeschini
        "New similarity coefficients for binary data"
        MATCH Commun.Math.Comput.Chem.. 68. 581-592.
        <https://match.pmf.kg.ac.rs/electronic_versions/Match68/n2/match68n2_581-592.pdf>`_

    .. [2] `Todeschini, Roberto, Davide Ballabio, and Viviana Consonni
        "Distances and similarity measures in chemometrics and chemoinformatics."
        Encyclopedia of Analytical Chemistry: Applications, Theory and Instrumentation (2006): 1-40.
        <https://doi.org/10.1002/9780470027318.a9438.pub2>`_

    .. [3] `Todeschini, Roberto, et al.
        "Similarity coefficients for binary chemoinformatics data: overview and
        extended comparison using simulated and real data sets."
        J. Chem. Inf. Model. 2012, 52, 11, 2884–2901
        <https://doi.org/10.1021/ci300261r>`_

    Examples
    --------
    >>> from skfp.distances import ct4_count_similarity
    >>> import numpy as np
    >>> vec_a = np.array([7, 1, 1])
    >>> vec_b = np.array([7, 1, 2])
    >>> sim = ct4_count_similarity(vec_a, vec_b)
    >>> sim
    0.9953140617275088

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[7, 1, 1]])
    >>> vec_b = csr_array([[7, 1, 2]])
    >>> sim = ct4_count_similarity(vec_a, vec_b)
    >>> sim
    0.9953140617275088
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

    numerator = np.log(1 + dot_ab)
    denominator = np.log(1 + dot_aa + dot_bb - dot_ab)

    sim = float(numerator / denominator) if denominator >= 1e-8 else 1.0

    return sim


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def ct4_count_distance(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    """
    Consonni–Todeschini distance for vectors of count values.

    Computes the Consonni–Todeschini 4 (CT4) distance [1]_ [2]_ [3]_ for count data
    between two input arrays or sparse matrices by subtracting the similarity
    from 1, using the formula:

    .. math::

        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`ct4_count_similarity`.
    The calculated distance falls within the range :math:`[0, 1]`.
    Vectors with 0 or 1 elements in their intersection or union (which
    would cause numerical problems with logarithm) have distance 1.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First count input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second count input array or sparse matrix.

    References
    ----------
    .. [1] `V. Consonni, R. Todeschini
        "New similarity coefficients for binary data"
        MATCH Commun.Math.Comput.Chem.. 68. 581-592.
        <https://match.pmf.kg.ac.rs/electronic_versions/Match68/n2/match68n2_581-592.pdf>`_

    .. [2] `Todeschini, Roberto, Davide Ballabio, and Viviana Consonni
        "Distances and similarity measures in chemometrics and chemoinformatics."
        Encyclopedia of Analytical Chemistry: Applications, Theory and Instrumentation (2006): 1-40.
        <https://doi.org/10.1002/9780470027318.a9438.pub2>`_

    .. [3] `Todeschini, Roberto, et al.
        "Similarity coefficients for binary chemoinformatics data: overview and
        extended comparison using simulated and real data sets."
        J. Chem. Inf. Model. 2012, 52, 11, 2884–2901
        <https://doi.org/10.1021/ci300261r>`_

    Returns
    -------
    distance : float
        CT4 distance between ``vec_a`` and ``vec_b``.

    Examples
    --------
    >>> from skfp.distances import ct4_count_distance
    >>> import numpy as np
    >>> vec_a = np.array([7, 1, 1])
    >>> vec_b = np.array([7, 1, 2])
    >>> dist = ct4_count_distance(vec_a, vec_b)
    >>> dist
    0.004685938272491197

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[7, 1, 1]])
    >>> vec_b = csr_array([[7, 1, 2]])
    >>> dist = ct4_count_distance(vec_a, vec_b)
    >>> dist
    0.004685938272491197
    """
    return 1 - ct4_count_similarity(vec_a, vec_b)


@validate_params(
    {"X": ["array-like"], "Y": ["array-like", None]},
    prefer_skip_nested_validation=True,
)
def bulk_ct4_binary_similarity(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    Bulk Consonni–Todeschini 4 similarity for binary matrices.

    Computes the pairwise Consonni–Todeschini 4 (CT4) similarity between
    binary matrices. If one array is passed, similarities are computed
    between its rows. For two arrays, similarities are between their respective
    rows, with `i`-th row and `j`-th column in output corresponding to `i`-th row
    from first array and `j`-th row from second array.

    See also :py:func:`ct4_binary_similarity`.

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
        Array with pairwise Consonni–Todeschini similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`ct4_binary_similarity` : Consonni–Todeschini similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_ct4_binary_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_ct4_binary_similarity(X, Y)
    >>> sim
    array([[1.        , 0.5       ],
           [0.63092975, 0.63092975]])
    """
    if Y is None:
        return _bulk_ct4_binary_similarity_single(X)
    else:
        return _bulk_ct4_binary_similarity_two(X, Y)


@numba.njit(parallel=True)
def _bulk_ct4_binary_similarity_single(X: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    sims = np.empty((m, m))

    # upper triangle - actual similarities
    for i in numba.prange(m):
        for j in numba.prange(i + 1, m):
            intersection = np.sum(np.logical_and(X[i], X[j]))
            union = np.sum(np.logical_or(X[i], X[j]))
            sims[i, j] = (
                float(np.log(1 + intersection) / np.log(1 + union))
                if union != 0
                else 1.0
            )

    # diagonal - always 1
    for i in numba.prange(m):
        sims[i, i] = 1.0

    # lower triangle - symmetric with upper triangle
    for i in numba.prange(1, m):
        for j in numba.prange(i):
            sims[i, j] = sims[j, i]

    return sims


@numba.njit(parallel=True)
def _bulk_ct4_binary_similarity_two(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    n = Y.shape[0]
    sims = np.empty((m, n))

    for i in numba.prange(m):
        for j in numba.prange(n):
            intersection = np.sum(np.logical_and(X[i], Y[j]))
            union = np.sum(np.logical_or(X[i], Y[j]))
            sims[i, j] = (
                float(np.log(1 + intersection) / np.log(1 + union))
                if union != 0
                else 1.0
            )

    return sims


@validate_params(
    {
        "X": ["array-like", csr_array],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_ct4_binary_distance(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    Bulk Consonni–Todeschini 4 distance for vectors of binary values.

    Computes the pairwise Consonni–Todeschini 4 (CT4) distance between binary matrices.
    If one array is passed, distances are computed between its rows. For two arrays,
    distances are between their respective rows, with `i`-th row and `j`-th column
    in output corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`ct4_binary_distance`.

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
        Array with pairwise Consonni–Todeschini distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`ct4_binary_distance` : Consonni–Todeschini distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_ct4_binary_distance
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_ct4_binary_distance(X, Y)
    >>> dist
    array([[0., 0.],
           [0., 0.]])

    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_ct4_binary_distance(X)
    >>> dist
    array([[0., 0.],
           [0., 0.]])
    """
    return 1 - bulk_ct4_binary_similarity(X, Y)


@validate_params(
    {"X": ["array-like"], "Y": ["array-like", None]},
    prefer_skip_nested_validation=True,
)
def bulk_ct4_count_similarity(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    Bulk Consonni–Todeschini 4 similarity for count matrices.

    Computes the pairwise Consonni–Todeschini 4 similarity between count matrices.
    If one array is passed, similarities are computed between its rows. For two arrays,
    similarities are between their respective rows, with `i`-th row and `j`-th column
    in output corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`ct4_count_similarity`.

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
        Array with pairwise Consonni–Todeschini similarity values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`ct4_count_similarity` : Consonni–Todeschini similarity function for two vectors.

    Examples
    --------
    >>> from skfp.distances import bulk_ct4_count_similarity
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [0, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [0, 1, 1]])
    >>> sim = bulk_ct4_count_similarity(X, Y)
    >>> sim
    array([[1.        , 0.5       ],
           [0.63092975, 0.63092975]])
    """
    X = X.astype(float)  # Numba does not allow integers

    if Y is None:
        return _bulk_ct4_count_similarity_single(X)
    else:
        Y = Y.astype(float)
        return _bulk_ct4_count_similarity_two(X, Y)


@numba.njit(parallel=True)
def _bulk_ct4_count_similarity_single(X: np.ndarray) -> np.ndarray:
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

            numerator = np.log(1 + dot_ab)
            denominator = np.log(1 + dot_aa + dot_bb - dot_ab)

            sims[i, j] = float(numerator / denominator) if denominator >= 1e-8 else 1.0

    # diagonal - always 1
    for i in numba.prange(m):
        sims[i, i] = 1.0

    # lower triangle - symmetric with upper triangle
    for i in numba.prange(1, m):
        for j in numba.prange(i):
            sims[i, j] = sims[j, i]

    return sims


@numba.jit(parallel=True)
def _bulk_ct4_count_similarity_two(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    n = Y.shape[0]
    sims = np.empty((m, n))

    for i in numba.prange(m):
        for j in numba.prange(n):
            vec_a = X[i]
            vec_b = Y[j]

            dot_aa = np.dot(vec_a, vec_a)
            dot_bb = np.dot(vec_b, vec_b)
            dot_ab = np.dot(vec_a, vec_b)

            numerator = np.log(1 + dot_ab)
            denominator = np.log(1 + dot_aa + dot_bb - dot_ab)

            sims[i, j] = float(numerator / denominator) if denominator >= 1e-8 else 1.0

    return sims


@validate_params(
    {
        "X": ["array-like", csr_array],
        "Y": ["array-like", csr_array, None],
    },
    prefer_skip_nested_validation=True,
)
def bulk_ct4_count_distance(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    Bulk Consonni–Todeschini 4 distance for vectors of count values.

    Computes the pairwise Consonni–Todeschini 4 distance between count matrices.
    If one array is passed, distances are computed between its rows. For two arrays,
    distances are between their respective rows, with `i`-th row and `j`-th column
    in output corresponding to `i`-th row from first array and `j`-th row from second array.

    See also :py:func:`ct4_count_distance`.

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
        Array with pairwise Consonni–Todeschini distance values. Shape is :math:`m \times n` if two
        arrays are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`ct4_count_distance` : Consonni–Todeschini distance function for two vectors

    Examples
    --------
    >>> from skfp.distances import bulk_ct4_count_distance
    >>> import numpy as np
    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> Y = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_ct4_count_distance(X, Y)
    >>> dist
    array([[0., 0.],
           [0., 0.]])

    >>> X = np.array([[1, 0, 1], [1, 0, 1]])
    >>> dist = bulk_ct4_count_distance(X)
    >>> dist
    array([[0., 0.],
           [0., 0.]])
    """
    return 1 - bulk_ct4_count_similarity(X, Y)
