from typing import Union

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
def sokal_sneath_2_binary_similarity(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    r"""
    Sokal-Sneath similarity 2 for vectors of binary values.

    Computes the Sokal-Sneath similarity 2 [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices, using the formula:

    .. math::

        sim(a, b) = \frac{|a \cap b|}{|a \cup b| + |a \Delta b} =
                    \frac{|a \cap b|}{2 * |a| + 2 * |b| - 3 * |a \cap b|}

    where :`|a \Delta b|` is the XOR operation (symmetric difference), i.e. number
    of bits that are "on" in one vector and "off" in another.

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
        Sokal-Sneath similarity 2 between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `R. R. Sokal, P. H. A. Sneath
        "Principles of Numerical Taxonomy"
        Principles of Numerical Taxonomy., 1963, 359 ref. bibl. 18 pp.
        <https://www.cabidigitallibrary.org/doi/full/10.5555/19650300280>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import sokal_sneath_2_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 1, 1, 1])
    >>> vec_b = np.array([1, 1, 0, 0])
    >>> sim = sokal_sneath_2_binary_similarity(vec_a, vec_b)
    >>> sim
    0.3333333333333333

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 1, 1, 1]])
    >>> vec_b = csr_array([[1, 1, 0, 0]])
    >>> sim = sokal_sneath_2_binary_similarity(vec_a, vec_b)
    >>> sim
    0.3333333333333333
    """
    if type(vec_a) is not type(vec_b):
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, "
            f"got {type(vec_a)} and {type(vec_b)}"
        )

    if isinstance(vec_a, np.ndarray):
        intersection = np.sum(np.logical_and(vec_a, vec_b))
        a_sum = np.sum(vec_a)
        b_sum = np.sum(vec_b)
    else:
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)

        intersection = len(vec_a_idxs & vec_b_idxs)
        a_sum = len(vec_a_idxs)
        b_sum = len(vec_b_idxs)

    denominator = 2 * a_sum + 2 * b_sum - 3 * intersection
    sim = intersection / denominator if denominator > 0 else 1.0

    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def sokal_sneath_2_binary_distance(
    vec_a: Union[np.ndarray, csr_array],
    vec_b: Union[np.ndarray, csr_array],
) -> float:
    """
    Sokal-Sneath distance 2 for vectors of binary values.

    Computes the Sokal-Sneath distance 2 [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices by subtracting the similarity from 1,
    using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`sokal_sneath_2_binary_similarity`.
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
        Sokal-Sneath distance 2 between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `R. R. Sokal, P. H. A. Sneath
        "Principles of Numerical Taxonomy"
        Principles of Numerical Taxonomy., 1963, 359 ref. bibl. 18 pp.
        <https://www.cabidigitallibrary.org/doi/full/10.5555/19650300280>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import sokal_sneath_2_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 1, 1, 1])
    >>> vec_b = np.array([1, 1, 0, 0])
    >>> dist = sokal_sneath_2_binary_distance(vec_a, vec_b)
    >>> dist
    0.6666666666666667

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 1, 1, 1]])
    >>> vec_b = csr_array([[1, 1, 0, 0]])
    >>> dist = sokal_sneath_2_binary_distance(vec_a, vec_b)
    >>> dist
    0.6666666666666667
    """
    return 1 - sokal_sneath_2_binary_similarity(vec_a, vec_b)
