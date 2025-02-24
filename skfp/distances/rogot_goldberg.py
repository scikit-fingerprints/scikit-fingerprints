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
def rogot_goldberg_binary_similarity(
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
    Rogot-Goldberg similarity for vectors of binary values.

    Computes the Rogot-Goldberg similarity [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices, using the formula:

    .. math::

        sim(x, y) = \frac{a}{2 * (2a + b + c)} +
                    \frac{d}{2 * (2d + b + c)}

    where :math:`a`, :math:`b`, :math:`c` and :math:`d` correspond to the number
    of bit relations between the two vectors:

    - :math:`a` - both are 1 (:math:`|x \cap y|`, common "on" bits)
    - :math:`b` - :math:`x` is 1, :math:`y` is 0
    - :math:`c` - :math:`x` is 0, :math:`y` is 1
    - :math:`d` - both are 0

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    -------
    similarity : float
        Rogot-Goldberg similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Rogot E., Goldberg I.D.
        "A proposed index for measuring agreement in test-retest studies."
        Journal of Chronic Diseases 19.9 (1966): 991-1006.`
        <https://doi.org/10.1016/0021-9681(66)90032-4>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import rogot_goldberg_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = rogot_goldberg_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = rogot_goldberg_binary_similarity(vec_a, vec_b)
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
        d = np.sum(~vec_a & ~vec_b)
    else:
        vec_a = vec_a.tocsr()
        vec_b = vec_b.tocsr()

        length = vec_a.shape[1]
        vec_a_idxs = set(vec_a.indices)
        vec_b_idxs = set(vec_b.indices)

        a = len(vec_a_idxs & vec_b_idxs)
        b = len(vec_a_idxs - vec_b_idxs)
        c = len(vec_b_idxs - vec_a_idxs)
        d = length - (a + b + c)

    first_denom = 2 * a + b + c
    second_denom = 2 * d + b + c

    # The denominator of the second part of the equation is equal to 1 if both vectors are all of 1s.
    if second_denom == 1:
        return 1.0

    rogot_goldberg_sim = a / first_denom + d / second_denom

    return float(rogot_goldberg_sim)


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
def rogot_goldberg_binary_distance(
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
    Rogot-Goldberg distance for vectors of binary values.

    Computes the Rogot-Goldberg distance [1]_ [2]_ [3]_ for binary data between two
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
        Rogot-Goldberg distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Rogot E., Goldberg I.D.
        "A proposed index for measuring agreement in test-retest studies."
        Journal of chronic diseases 19.9 (1966): 991-1006.`
        <https://doi.org/10.1016/0021-9681(66)90032-4>`_

    .. [2] `Deza M.M., Deza E.
        "Encyclopedia of Distances."
        Springer, Berlin, Heidelberg, 2009.
        <https://doi.org/10.1007/978-3-642-00234-2_1>`_

    .. [3] `RDKit documentation
        <https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html>`_

    Examples
    --------
    >>> from skfp.distances import rogot_goldberg_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = rogot_goldberg_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = rogot_goldberg_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - rogot_goldberg_binary_similarity(vec_a, vec_b)
