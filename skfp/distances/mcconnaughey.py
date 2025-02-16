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
def mcconnaughey_binary_similarity(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
    normalized: bool = True,
) -> float:
    r"""
    McConnaughey similarity for vectors of binary values.

    Computes the McConnaughey similarity [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{(|a \cap b \cdot (|a| + |b|) - |a| \cdot |b|}{|a| \cdot |b|}


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

    normalized : bool, default=True
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
    _check_finite_values(vec_a)
    _check_finite_values(vec_b)
    _check_valid_vectors(vec_a, vec_b)

    if np.sum(vec_a) == 0 == np.sum(vec_b):
        return 1.0

    if isinstance(vec_a, np.ndarray):
        num_common = np.sum(np.logical_and(vec_a, vec_b))
        vec_a_ones = np.sum(vec_a)
        vec_b_ones = np.sum(vec_b)

    else:
        num_common = len(set(vec_a.indices) & set(vec_b.indices))
        vec_a_ones = len(set(vec_a.indices))
        vec_b_ones = len(set(vec_b.indices))

    if vec_a_ones * vec_b_ones == 0:
        return -1 if not normalized else 0

    sim = (num_common * (vec_a_ones + vec_b_ones) - (vec_a_ones * vec_b_ones)) / (
        vec_a_ones * vec_b_ones
    )

    if normalized:
        sim = (sim + 1) / 2

    return sim


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def mcconnaughey_binary_distance(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
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
