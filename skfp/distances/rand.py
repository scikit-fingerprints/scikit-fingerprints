from typing import Union

import numpy as np
from scipy.sparse import csc_array, csr_array
from sklearn.utils._param_validation import validate_params


@validate_params(
    {
        "vec_a": ["array-like", csr_array, csc_array],
        "vec_b": ["array-like", csr_array, csc_array],
    },
    prefer_skip_nested_validation=True,
)
def rand_binary_similarity(
    vec_a: Union[np.ndarray, csr_array, csc_array],
    vec_b: Union[np.ndarray, csr_array, csc_array],
) -> float:
    r"""
    Rand similarity for vectors of binary values.

    Computes the Rand similarity [1]_ [2]_ (known as All-Bit [3]_ or Sokal-Michener)
    for binary data between two input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{|a \cap b|}{n}

    where `n` is the length of vector `a`.

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
        Rand similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Rand, W.M.
        "Objective criteria for the evaluation of clustering methods."
        J. Amer. Stat. Assoc. 1971; 66: 846–850.
        <https://www.tandfonline.com/doi/abs/10.1080/01621459.1971.10482356>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import rand_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = rand_binary_similarity(vec_a, vec_b)
    >>> sim
    0.6666666666666666

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = rand_binary_similarity(vec_a, vec_b)
    >>> sim
    0.6666666666666666
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if isinstance(vec_a, np.ndarray):
        num_common = np.sum(np.logical_and(vec_a, vec_b))
        length = len(vec_a)
    else:
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)
        num_common = len(vec_a_idxs & vec_b_idxs)
        length = vec_a.shape[1]

    rand_sim = num_common / length
    return float(rand_sim)


@validate_params(
    {
        "vec_a": ["array-like", csr_array, csc_array],
        "vec_b": ["array-like", csr_array, csc_array],
    },
    prefer_skip_nested_validation=True,
)
def rand_binary_distance(
    vec_a: Union[np.ndarray, csr_array, csc_array],
    vec_b: Union[np.ndarray, csr_array, csc_array],
) -> float:
    """
    Rand distance for vectors of binary values.

    Computes the Rand distance [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices by subtracting the similarity from 1,
    using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`rand_binary_similarity`.
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
        Rand distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Rand, W.M.
        "Objective criteria for the evaluation of clustering methods."
        J. Amer. Stat. Assoc. 1971; 66: 846–850.
        <https://www.tandfonline.com/doi/abs/10.1080/01621459.1971.10482356>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import rand_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = rand_binary_distance(vec_a, vec_b)
    >>> dist
    0.33333333333333337

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = rand_binary_distance(vec_a, vec_b)
    >>> dist
    0.33333333333333337
    """
    return 1 - rand_binary_similarity(vec_a, vec_b)
