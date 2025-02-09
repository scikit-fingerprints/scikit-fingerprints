from typing import Union

import numpy as np
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
def rand_binary_similarity(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
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
    1.0

    >>> from skfp.distances import rand_binary_similarity
    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = rand_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0
    """
    _check_finite_values(vec_a)
    _check_finite_values(vec_b)
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Vectors must have same shape, got {vec_a.shape} and {vec_b.shape}"
        )

    if np.sum(vec_a) == 0 == np.sum(vec_b):
        return 0.0

    if isinstance(vec_a, np.ndarray) and isinstance(vec_b, np.ndarray):
        length = len(vec_a)
        n_equal_vals = np.sum(vec_a == vec_b)
    elif isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        length = vec_a.shape[1]
        n_equal_vals = length - (vec_a != vec_b).nnz
    else:
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, either numpy.ndarray "
            f"or scipy.sparse.csr_array, got {type(vec_a)} and {type(vec_b)}"
        )

    rand_sim = n_equal_vals / length
    return float(rand_sim)


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def rand_binary_distance(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
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
    0.0

    >>> from skfp.distances import rand_binary_distance
    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = rand_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - rand_binary_similarity(vec_a, vec_b)
