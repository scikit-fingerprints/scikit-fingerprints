from typing import Union

import numpy as np
from numba import njit, prange
from scipy.sparse import csr_array
from sklearn.utils._param_validation import validate_params

from .utils import _check_finite_values


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def ct4_binary_similarity(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    r"""
    Consonni–Todeschini 4 similarity for vectors of binary values.

    Computes the Consonni–Todeschini 4 similarity [1]_ [2]_ [3]_ for binary data
    between two input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{\log |a \cap b|}{\log |a \cup b|}
        = \frac{\log |a \cap b|}{\log (|a| + |b| - |a \cap b|)}

    It is calculated by log-transforming numerator and denominator of the
    Tanimoto similarity (see also :py:func:`tanimoto_binary_similarity`).

    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a similarity of 1.
    Vectors with 0 common elements or only 1 bit with value 1, which would
    result in wrong logarithm values, have similarity of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        CT4 similarity between vec_a and vec_b.

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

    >>> from skfp.distances import ct4_binary_similarity
    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = ct4_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0
    """
    _check_finite_values(vec_a)
    _check_finite_values(vec_b)

    if np.sum(vec_a) == 0 == np.sum(vec_b):
        return 1.0

    if isinstance(vec_a, np.ndarray) and isinstance(vec_b, np.ndarray):
        intersection = np.sum(vec_a & vec_b)
        union = np.sum(vec_a | vec_b)
    elif isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        intersection = vec_a.multiply(vec_b).sum()
        union = vec_a.sum() + vec_b.sum() - intersection
    else:
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, either numpy.ndarray "
            f"or scipy.sparse.csr_array, got {type(vec_a)} and {type(vec_b)}"
        )

    if intersection == 0 or union == 1:
        # log of 0 is -infinity, and log of 1 is 0
        return 0.0

    return float(np.log(intersection) / np.log(union))


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def ct4_binary_distance(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    """
    Consonni–Todeschini distance for vectors of binary values.

    Computes the Consonni–Todeschini 4 distance [1]_ [2]_ [3]_ for binary data
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

    >>> from skfp.distances import ct4_binary_distance
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

    Computes the Consonni–Todeschini 4 similarity [1]_ [2]_ [3]_ for count data
    between two input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{\log (a \cdot b)}{\log (\|a\|^2 + \|b\|^2 - a \cdot b)}

    It is calculated by log-transforming numerator and denominator of the
    CT4 count similarity (see also :py:func:`ct4_count_similarity`).

    Calculated similarity falls within the range of :math:`[0, 1]`.
    Passing all-zero vectors to this function results in similarity of 1.

    Note that Numpy version is optimized with Numba JIT compiler, resulting in
    significantly faster performance compared to SciPy sparse arrays. First usage
    may be slightly slower due to Numba compilation.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First count input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second count input array or sparse matrix.

    Returns
    -------
    similarity : float
        CT4 similarity between vec_a and vec_b.

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
    0.9952023187751823

    >>> from skfp.distances import ct4_count_similarity
    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[7, 1, 1]])
    >>> vec_b = csr_array([[7, 1, 2]])
    >>> sim = ct4_count_similarity(vec_a, vec_b)
    >>> sim
    0.9952023187751823
    """
    _check_finite_values(vec_a)
    _check_finite_values(vec_b)

    if np.sum(vec_a) == 0 and np.sum(vec_b) == 0:
        return 1.0

    if isinstance(vec_a, np.ndarray) and isinstance(vec_b, np.ndarray):
        intersection, union = _ct4_count_numpy(vec_a, vec_b)
    elif isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        intersection, union = _ct4_count_scipy(vec_a, vec_b)
    else:
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, either numpy.ndarray "
            f"or scipy.sparse.csr_array, got {type(vec_a)} and {type(vec_b)}"
        )

    if np.isclose(intersection, 0) or np.isclose(union, 1):
        # log of 0 is -infinity, and log of 1 is 0
        return 0.0

    return float(np.log(intersection) / np.log(union))


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

    Computes the Consonni–Todeschini 4 distance [1]_ [2]_ [3]_ for count data
    between two input arrays or sparse matrices by subtracting the similarity
    from 1, using the formula:

    .. math::

        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`ct4_count_similarity`.
    The calculated distance falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a distance of 0.
    Vectors with 0 common elements or only 1 bit with value 1, which would
    result in wrong logarithm values, have distance of 1.

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
    0.004797681224817718

    >>> from skfp.distances import ct4_count_distance
    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[7, 1, 1]])
    >>> vec_b = csr_array([[7, 1, 2]])
    >>> dist = ct4_count_distance(vec_a, vec_b)
    >>> dist
    0.004797681224817718
    """
    return 1 - ct4_count_similarity(vec_a, vec_b)


@njit(parallel=True)
def _ct4_count_numpy(vec_a: np.ndarray, vec_b: np.ndarray) -> tuple[float, float]:
    vec_a = vec_a.astype(np.float64).ravel()
    vec_b = vec_b.astype(np.float64).ravel()

    dot_ab = 0.0
    dot_aa = 0.0
    dot_bb = 0.0

    for i in prange(vec_a.shape[0]):
        dot_ab += vec_a[i] * vec_b[i]
        dot_aa += vec_a[i] * vec_a[i]
        dot_bb += vec_b[i] * vec_b[i]

    intersection = dot_ab
    union = dot_aa + dot_bb - dot_ab

    return intersection, union


def _ct4_count_scipy(vec_a: csr_array, vec_b: csr_array) -> tuple[float, float]:
    dot_ab: float = vec_a.multiply(vec_b).sum()
    dot_aa: float = vec_a.multiply(vec_a).sum()
    dot_bb: float = vec_b.multiply(vec_b).sum()

    intersection = dot_ab
    union = dot_aa + dot_bb - dot_ab

    return intersection, union
