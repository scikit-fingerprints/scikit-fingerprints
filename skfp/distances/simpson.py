from typing import Union

import numpy as np
from scipy.sparse import csr_array
from sklearn.utils._param_validation import validate_params

from .utils import _check_finite_values, _check_valid_vectors


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def simpson_binary_similarity(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    r"""
    Simpson similarity for vectors of binary values.

    Computes the Simpson similarity [1]_ (also known as asymmetric similarity [2]_ [3]_
    or overlap coefficient [4]_) for binary data between two input arrays or sparse
    matrices using the formula:

    .. math::

        sim(a, b) = \frac{|a \cap b|}{\min(|a|, |b|)}

    The calculated similarity falls within the range :math:`[0, 1]`.
    If any of the vectors is all-zeros, it results in a similarity of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        Simpson similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Simpson, G.G.
       "Mammals and the nature of continents."
       American Journal of Science, 241: 1-31 (1943).
       <https://doi.org/10.1038/163688a0>`_

    .. [2] `Deza M.M., Deza E.
       "Encyclopedia of Distances."
       Springer, Berlin, Heidelberg, 2009.
       <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
       <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    .. [4] `Overlap coefficient on Wikipedia
       <https://en.wikipedia.org/wiki/Overlap_coefficient>`_

    Examples
    --------
    >>> from skfp.distances import simpson_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = simpson_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = simpson_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0
    """
    _check_finite_values(vec_a)
    _check_finite_values(vec_b)
    _check_valid_vectors(vec_a, vec_b)

    if np.sum(vec_a) == 0 or np.sum(vec_b) == 0:
        return 0.0

    if isinstance(vec_a, np.ndarray):
        num_common = np.sum(np.logical_and(vec_a, vec_b))
    else:
        num_common = len(set(vec_a.indices) & set(vec_b.indices))

    min_vec = min(np.sum(vec_a), np.sum(vec_b))

    braun_blanquet_sim = num_common / min_vec
    return float(braun_blanquet_sim)


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def simpson_binary_distance(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    """
    Simpson distance for vectors of binary values.

    Computes the Simpson distance for binary data between two input arrays
    or sparse matrices by subtracting the Simpson similarity [1]_ [2]_ [3]_ [4]_
    from 1, using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`simpson_binary_similarity`.
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
        Simpson distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Simpson, G.G.
       "Mammals and the nature of continents."
       American Journal of Science, 241: 1-31 (1943).
       <https://doi.org/10.1038/163688a0>`_

    .. [2] `Deza M.M., Deza E.
       "Encyclopedia of Distances."
       Springer, Berlin, Heidelberg, 2009.
       <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
       <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    .. [4] `Overlap coefficient on Wikipedia
       <https://en.wikipedia.org/wiki/Overlap_coefficient>`_

    Examples
    --------
    >>> from skfp.distances import simpson_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = simpson_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = simpson_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - simpson_binary_similarity(vec_a, vec_b)
