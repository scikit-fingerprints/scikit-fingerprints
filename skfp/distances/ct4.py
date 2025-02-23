from typing import Union

import numpy as np
from scipy.sparse import coo_array, csc_array, csr_array
from sklearn.utils._param_validation import validate_params

from .utils import _check_finite_values, _check_valid_vectors


@validate_params(
    {
        "vec_a": [
            "array-like",
            csr_array,
            coo_array,
            csc_array,
        ],
        "vec_b": [
            "array-like",
            csr_array,
            coo_array,
            csc_array,
        ],
    },
    prefer_skip_nested_validation=True,
)
def ct4_binary_similarity(
    vec_a: Union[
        np.ndarray,
        csr_array,
        coo_array,
        csc_array,
    ],
    vec_b: Union[
        np.ndarray,
        csr_array,
        coo_array,
        csc_array,
    ],
) -> float:
    r"""
    Consonni–Todeschini 4 similarity for vectors of binary values.

    Computes the Consonni–Todeschini 4 similarity [1]_ [2]_ [3]_ for binary data
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
    _check_finite_values(vec_a)
    _check_finite_values(vec_b)
    _check_valid_vectors(vec_a, vec_b)

    if np.sum(vec_a) == 0 == np.sum(vec_b):
        return 1.0

    if isinstance(vec_a, np.ndarray):
        intersection = np.sum(np.logical_and(vec_a, vec_b))
        union = np.sum(np.logical_or(vec_a, vec_b))
    else:
        vec_a = vec_a.tocsr()
        vec_b = vec_b.tocsr()

        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)
        intersection = len(vec_a_idxs & vec_b_idxs)
        union = len(vec_a_idxs | vec_b_idxs)

    return float(np.log(1 + intersection) / np.log(1 + union))


@validate_params(
    {
        "vec_a": [
            "array-like",
            csr_array,
            coo_array,
            csc_array,
        ],
        "vec_b": [
            "array-like",
            csr_array,
            coo_array,
            csc_array,
        ],
    },
    prefer_skip_nested_validation=True,
)
def ct4_binary_distance(
    vec_a: Union[
        np.ndarray,
        csr_array,
        coo_array,
        csc_array,
    ],
    vec_b: Union[
        np.ndarray,
        csr_array,
        coo_array,
        csc_array,
    ],
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
        "vec_a": [
            "array-like",
            csr_array,
            coo_array,
            csc_array,
        ],
        "vec_b": [
            "array-like",
            csr_array,
            coo_array,
            csc_array,
        ],
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
    _check_finite_values(vec_a)
    _check_finite_values(vec_b)
    _check_valid_vectors(vec_a, vec_b)

    if np.sum(vec_a) == 0 == np.sum(vec_b):
        return 1.0

    if isinstance(vec_a, np.ndarray):
        dot_aa = np.dot(vec_a, vec_a)
        dot_bb = np.dot(vec_b, vec_b)
        dot_ab = np.dot(vec_a, vec_b)
    else:
        dot_ab = vec_a.multiply(vec_b).sum()
        dot_aa = vec_a.multiply(vec_a).sum()
        dot_bb = vec_b.multiply(vec_b).sum()

    intersection = 1 + dot_ab
    union = 1 + dot_aa + dot_bb - dot_ab

    return float(np.log(intersection) / np.log(union))


@validate_params(
    {
        "vec_a": [
            "array-like",
            csr_array,
            coo_array,
            csc_array,
        ],
        "vec_b": [
            "array-like",
            csr_array,
            coo_array,
            csc_array,
        ],
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
