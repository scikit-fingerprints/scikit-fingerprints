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
def harris_lahey_binary_similarity(
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
    normalized: bool = False,
) -> float:
    r"""
    Harris-Lahey similarity for vectors of binary values.

    Computes the Harris-Lahey similarity [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices, using the formula:

    .. math::

        sim(x, y) = \frac{a * (2d + b + c)}{2 * (a + b + c)} +
                    \frac{d * (2a + b + c)}{2 * (b + c + d)}

    where :math:`a`, :math:`b`, :math:`c` and :math:`d` correspond to the number
    of bit relations between the two vectors:

    - :math:`a` - both are 1 (:math:`|x \cap y|`, common "on" bits)
    - :math:`b` - :math:`x` is 1, :math:`y` is 0
    - :math:`c` - :math:`x` is 0, :math:`y` is 1
    - :math:`d` - both are 0

    The calculated similarity falls within the range :math:`[0, n]`, where :math:`n`
    is the length of vectors. Use ``normalized`` argument to scale the similarity to
    range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a similarity of 1.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    normalized : bool, default=False
        Whether to divide the resulting similarity by length of vectors (their number
        of elements), to normalize values to range ``[0, 1]``.

    Returns
    -------
    similarity : float
        Harris-Lahey similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Francis C. Harris, Benjamin B. Lahey
        "A method for combining occurrence and nonoccurrence interobserver agreement scores"
        J Appl Behav Anal. 1978 Winter;11(4):523-7.
        <https://doi.org/10.1901/jaba.1978.11-523>`_

    .. [2] `Brusco M, Cradit JD, Steinley D.
        "A comparison of 71 binary similarity coefficients: The effect of base rates"
        PLoS One. 2021 Apr 7;16(4):e0247751.
        <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0247751>`_

    .. [3] `R. Todeschini, V. Consonni, H. Xiang, J. Holliday, M. Buscema, P. Willett
        "Similarity Coefficients for Binary Chemoinformatics Data: Overview and
        Extended Comparison Using Simulated and Real Data Sets"
        J. Chem. Inf. Model. 2012, 52, 11, 2884–2901
        <https://doi.org/10.1021/ci300261r>`_

    Examples
    --------
    >>> from skfp.distances import harris_lahey_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = harris_lahey_binary_similarity(vec_a, vec_b)
    >>> sim
    3.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = harris_lahey_binary_similarity(vec_a, vec_b)
    >>> sim
    3.0
    """
    _check_finite_values(vec_a)
    _check_finite_values(vec_b)
    _check_valid_vectors(vec_a, vec_b)

    if np.sum(vec_a) == 0 == np.sum(vec_b):
        return 1.0

    if isinstance(vec_a, np.ndarray):
        length = len(vec_a)
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
        d = length - a

    first_elem = (a * (2 * d + b + c)) / (2 * (a + b + c))
    second_elem = (d * (2 * a + b + c)) / (2 * (b + c + d))
    sim = float(first_elem + second_elem)

    if normalized:
        sim = sim / length

    return sim


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
def harris_lahey_binary_distance(
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
    Harris-Lahey distance for vectors of binary values.

    Computes the Harris-Lahey distance [1]_ [2]_ [3]_ for binary data between two
    input arrays or sparse matrices by subtracting the similarity from 1,
    using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`harris_lahey_binary_similarity`. It uses the normalized
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
        Harris-Lahey distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] `Francis C. Harris, Benjamin B. Lahey
        "A method for combining occurrence and nonoccurrence interobserver agreement scores"
        J Appl Behav Anal. 1978 Winter;11(4):523-7.
        <https://doi.org/10.1901/jaba.1978.11-523>`_

    .. [2] `Brusco M, Cradit JD, Steinley D.
        "A comparison of 71 binary similarity coefficients: The effect of base rates"
        PLoS One. 2021 Apr 7;16(4):e0247751.
        <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0247751>`_

    .. [3] `R. Todeschini, V. Consonni, H. Xiang, J. Holliday, M. Buscema, P. Willett
        "Similarity Coefficients for Binary Chemoinformatics Data: Overview and
        Extended Comparison Using Simulated and Real Data Sets"
        J. Chem. Inf. Model. 2012, 52, 11, 2884–2901
        <https://doi.org/10.1021/ci300261r>`_

    Examples
    --------
    >>> from skfp.distances import harris_lahey_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = harris_lahey_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = harris_lahey_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """
    return 1 - harris_lahey_binary_similarity(vec_a, vec_b, normalized=True)
