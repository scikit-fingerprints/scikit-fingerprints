from typing import Union

import numpy as np
from scipy.sparse import coo_array, csc_array, csr_array
from sklearn.utils._param_validation import validate_params


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
def braun_blanquet_binary_similarity(
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
    Braun-Blanquet similarity for vectors of binary values.

    Computes the Braun-Blanquet similarity [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices using the formula:

    .. math::

        sim(a, b) = \frac{|a \cap b|}{\max(|a|, |b|)}

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
    similarity : float
        Braun-Blanquet similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Braun-Blanquet, J.
        "Plant sociology. The study of plant communities. First ed."
        McGraw-Hill Book Co., Inc., New York and London, 1932.
        <https://www.cabidigitallibrary.org/doi/full/10.5555/19331600801>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import braun_blanquet_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = braun_blanquet_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[0, 0, 1]])
    >>> sim = braun_blanquet_binary_similarity(vec_a, vec_b)
    >>> sim
    0.5
    """
    if type(vec_a) is not type(vec_b):
        raise ValueError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if np.sum(vec_a) == 0 == np.sum(vec_b):
        return 1.0

    if isinstance(vec_a, coo_array):
        vec_a = vec_a.tocsr()
        vec_b = vec_b.tocsr()

    if isinstance(vec_a, np.ndarray):
        num_common = np.sum(np.logical_and(vec_a, vec_b))
    else:
        num_common = len(set(vec_a.indices) & set(vec_b.indices))

    max_vec = max(np.sum(vec_a), np.sum(vec_b))

    braun_blanquet_sim = num_common / max_vec
    return float(braun_blanquet_sim)


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
def braun_blanquet_binary_distance(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    """
    Braun-Blanquet distance for vectors of binary values.

    Computes the Braun-Blanquet distance for binary data between two input arrays
    or sparse matrices by subtracting the Braun-Blanquet similarity [1]_ [2]_ [3]_
    from 1, using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`braun_blanquet_binary_similarity`.
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
        Braun-Blanquet distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Braun-Blanquet, J.
        "Plant sociology. The study of plant communities. First ed."
        McGraw-Hill Book Co., Inc., New York and London, 1932.
        <https://www.cabidigitallibrary.org/doi/full/10.5555/19331600801>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import braun_blanquet_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = braun_blanquet_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = braun_blanquet_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - braun_blanquet_binary_similarity(vec_a, vec_b)
