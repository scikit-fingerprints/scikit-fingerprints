from typing import Union

import numpy as np
from scipy.sparse import coo_array, csc_array, csr_array
from sklearn.utils._param_validation import validate_params

from .utils import _check_finite_values, _check_valid_vectors


@validate_params(
    {
        "vec_a": ["array-like", coo_array, csc_array, csr_array],
        "vec_b": ["array-like", coo_array, csc_array, csr_array],
    },
    prefer_skip_nested_validation=True,
)
def kulczynski_binary_similarity(
    vec_a: Union[np.ndarray, coo_array, csc_array, csr_array],
    vec_b: Union[np.ndarray, coo_array, csc_array, csr_array],
) -> float:
    r"""
    Kulczynski similarity for vectors of binary values.

    Computes the Kulczynski II similarity [1]_ [2]_ [3]_ for binary data between
    two input arrays or sparse matrices using the formula:

    .. math::

        sim(x, y) = \frac{1}{2} \left( \frac{a}{a+b} + \frac{a}{a+c} \right)

    where :math:`a`, :math:`b` and :math:`c` correspond to the number of bit
    relations between the two vectors:

    - :math:`a` - both are 1 (:math:`|x \cap y|`, common "on" bits)
    - :math:`b` - :math:`x` is 1, :math:`y` is 0
    - :math:`c` - :math:`x` is 0, :math:`y` is 1

    Note that this is the second Kulczynski similarity, also used by RDKit. It
    differs from Kulczynski I similarity from e.g. SciPy.

    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing two all-zero vectors to this function results in a similarity of 1.
    However, when only one is all-zero (i.e. :math:`a+b=0` or :math:`a+c=0`), the
    similarity is 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        Kulczynski similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] Kulczynski, S.
        "Zespoly roslin w Pieninach."
        Bull Int l’Academie Pol des Sci des Lettres (1927): 57-203.

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import kulczynski_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = kulczynski_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = kulczynski_binary_similarity(vec_a, vec_b)
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

        a = np.sum(vec_a & vec_b)
        b = np.sum(vec_a & ~vec_b)
        c = np.sum(~vec_a & vec_b)

    else:
        vec_a = vec_a.tocsr()
        vec_b = vec_b.tocsr()

        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)

        a = len(vec_a_idxs & vec_b_idxs)
        b = len(vec_a_idxs - vec_b_idxs)
        c = len(vec_b_idxs - vec_a_idxs)

    if a + b == 0 or a + c == 0:
        return 0.0

    sim = (a / (a + b) + a / (a + c)) / 2
    return float(sim)


@validate_params(
    {
        "vec_a": ["array-like", coo_array, csc_array, csr_array],
        "vec_b": ["array-like", coo_array, csc_array, csr_array],
    },
    prefer_skip_nested_validation=True,
)
def kulczynski_binary_distance(
    vec_a: Union[np.ndarray, coo_array, csc_array, csr_array],
    vec_b: Union[np.ndarray, coo_array, csc_array, csr_array],
) -> float:
    """
    Kulczynski distance for vectors of binary values.

    Computes the Kulczynski II distance for binary data between two input arrays
    or sparse matrices by subtracting the Kulczynski II similarity [1]_ [2]_ [3]_
    from 1, using to the formula:

    .. math::

        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`kulczynski_binary_similarity`.
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
        Kulczynski distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] Kulczynski, S.
        "Zespoly roslin w Pieninach."
        Bull Int l’Academie Pol des Sci des Lettres (1927): 57-203.

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import kulczynski_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = kulczynski_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = kulczynski_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - kulczynski_binary_similarity(vec_a, vec_b)
