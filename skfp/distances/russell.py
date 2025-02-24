from typing import Union

import numpy as np
from scipy.sparse import coo_array, csc_array, csr_array
from sklearn.utils._param_validation import validate_params

from .utils import _check_finite_values, _check_valid_vectors


@validate_params(
    {
        "vec_a": [
            "array-like",
            coo_array,
            csc_array,
            csr_array,
        ],
        "vec_b": [
            "array-like",
            coo_array,
            csc_array,
            csr_array,
        ],
    },
    prefer_skip_nested_validation=True,
)
def russell_binary_similarity(
    vec_a: Union[
        np.ndarray,
        coo_array,
        csc_array,
        csr_array,
    ],
    vec_b: Union[
        np.ndarray,
        coo_array,
        csc_array,
        csr_array,
    ],
) -> float:
    r"""
    Russell similarity for vectors of binary values.

    Computes the Russell similarity [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices, using the formula:

    .. math::

        sim(x, y) = \frac{a}{n}

    where

    - :math:`a` - common "on" bits.
    - :math:`n` - length of passed vectors

    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a similarity of 0.
    Passing empty vectors results in similarity of 0.

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
        "On habitat and association of species of anopheline larvae in south-eastern Madras. (1940)"
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
    >>> vec_a = np.array([1, 1, 1])
    >>> vec_b = np.array([1, 1, 1])
    >>> sim = russell_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 1, 1]])
    >>> vec_b = csr_array([[1, 1, 1]])
    >>> sim = russell_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0
    """
    _check_finite_values(vec_a)
    _check_finite_values(vec_b)
    _check_valid_vectors(vec_a, vec_b)

    if np.sum(vec_a) == 0 == np.sum(vec_b):
        return 1.0

    if isinstance(vec_a, np.ndarray):
        vec_a = vec_a.astype(bool)
        vec_b = vec_b.astype(bool)
        n = len(vec_a)
        a = np.sum(vec_a & vec_b)
    else:
        vec_a = vec_a.tocsr()
        vec_b = vec_b.tocsr()

        n = vec_a.shape[1]
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)

        a = len(vec_a_idxs & vec_b_idxs)

    # Empty vectors
    if n == 0:
        return 0.0

    russell_sim = a / n

    return float(russell_sim)


@validate_params(
    {
        "vec_a": [
            "array-like",
            coo_array,
            csc_array,
            csr_array,
        ],
        "vec_b": [
            "array-like",
            coo_array,
            csc_array,
            csr_array,
        ],
    },
    prefer_skip_nested_validation=True,
)
def russell_binary_distance(
    vec_a: Union[
        np.ndarray,
        coo_array,
        csc_array,
        csr_array,
    ],
    vec_b: Union[
        np.ndarray,
        coo_array,
        csc_array,
        csr_array,
    ],
) -> float:
    """
    Russell distance for vectors of binary values.

    Computes the Russell distance [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices by subtracting the similarity from 1,
    using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

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
        "On habitat and association of species of anopheline larvae in south-eastern Madras. (1940)"
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
    >>> vec_a = np.array([1, 1, 1])
    >>> vec_b = np.array([1, 1, 1])
    >>> dist = russell_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 1, 1]])
    >>> vec_b = csr_array([[1, 1, 1]])
    >>> dist = russell_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - russell_binary_similarity(vec_a, vec_b)
