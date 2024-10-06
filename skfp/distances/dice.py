from typing import Union

import numpy as np
from scipy.sparse import csr_array
from scipy.spatial.distance import dice
from sklearn.utils._param_validation import validate_params

from .utils import _check_nan


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def dice_binary_similarity(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    """
    Dice similarity for vectors of binary values.

    Computes the Dice similarity [1]_ [2]_ [3]_ for binary data between two input arrays
    or sparse matrices using the formula:

    .. math::

        sim(vec_a, vec_b) = \\frac{2 |vec_a \\cap vec_b|}{|vec_a| + |vec_b|}

    The calculated similarity falls within the range ``[0, 1]``.
    Passing all-zero vectors to this function results in a similarity of 1.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    ----------
    similarity : float
        Dice similarity between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] Dice, Lee R.
        "Measures of the amount of ecologic association between species."
        Ecology 26.3 (1945): 297-302.
        <https://esajournals.onlinelibrary.wiley.com/doi/abs/10.2307/1932409>

    .. [2] Brusco M., Cradit J. D., Steinley D.
        "A comparison of 71 binary similarity coefficients: The effect of base rates."
        PloS one 16.4 (2021): e0247751.
        <https://doi.org/10.1371/journal.pone.0247751>

    .. [3] Todeschini R., Consonni V., Xiang H., Holliday J., Buscema M., Willett P.
        "Similarity Coefficients for Binary Chemoinformatics Data: Overview and
        Extended Comparison Using Simulated and Real Data Sets."
        Journal of Chemical Information and Modeling 52.11 (2012): 2884-2901.
        <https://doi.org/10.1021/ci300261r>


    Examples
    ----------
    >>> from skfp.distances import dice_binary_similarity
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> sim = dice_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0

    >>> from skfp.distances import dice_binary_similarity
    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> sim = dice_binary_similarity(vec_a, vec_b)
    >>> sim
    1.0
    """
    _check_nan(vec_a)
    _check_nan(vec_b)

    if np.sum(vec_a) == 0 == np.sum(vec_b):
        return 1.0

    if isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        vec_a_bool = vec_a.astype(bool)
        vec_b_bool = vec_b.astype(bool)
        intersection: float = vec_a_bool.multiply(vec_b_bool).sum()
        return 2 * intersection / (vec_a_bool.sum() + vec_b_bool.sum())
    elif isinstance(vec_a, np.ndarray) and isinstance(vec_b, np.ndarray):
        vec_a_bool = vec_a.astype(bool)
        vec_b_bool = vec_b.astype(bool)
        return 1 - dice(vec_a_bool, vec_b_bool)
    else:
        raise TypeError(
            f"Both vec_a and vec_b must be of the same type, either numpy.ndarray "
            f"or scipy.sparse.csr_array, got {type(vec_a)} and {type(vec_b)}"
        )


@validate_params(
    {
        "vec_a": ["array-like", csr_array],
        "vec_b": ["array-like", csr_array],
    },
    prefer_skip_nested_validation=True,
)
def dice_binary_distance(
    vec_a: Union[np.ndarray, csr_array], vec_b: Union[np.ndarray, csr_array]
) -> float:
    """
    Dice distance for vectors of binary values.

    Computes the Dice distance for binary data between two input arrays
    or sparse matrices by subtracting the Dice similarity [1]_ [2]_ [3]_ from 1,
    using to the formula:

    .. math::

        dist(vec_a, vec_b) = 1 - sim(vec_a, vec_b)

    The calculated distance falls within the range ``[0, 1]``.
    Passing all-zero vectors to this function results in a distance of 0.

    Parameters
    ----------
    vec_a : {ndarray, sparse matrix}
        First binary input array or sparse matrix.

    vec_b : {ndarray, sparse matrix}
        Second binary input array or sparse matrix.

    Returns
    ----------
    distance : float
        Dice distance between ``vec_a`` and ``vec_b``.

    References
    ----------
    .. [1] Dice, Lee R.
        "Measures of the amount of ecologic association between species."
        Ecology 26.3 (1945): 297-302.
        <https://esajournals.onlinelibrary.wiley.com/doi/abs/10.2307/1932409>

    .. [2] Brusco M., Cradit J. D., Steinley D.
        "A comparison of 71 binary similarity coefficients: The effect of base rates."
        PloS one 16.4 (2021): e0247751.
        <https://doi.org/10.1371/journal.pone.0247751>

    .. [3] Todeschini R., Consonni V., Xiang H., Holliday J., Buscema M., Willett P.
        "Similarity Coefficients for Binary Chemoinformatics Data: Overview and
        Extended Comparison Using Simulated and Real Data Sets."
        Journal of Chemical Information and Modeling 52.11 (2012): 2884-2901.
        <https://doi.org/10.1021/ci300261r>

    Examples
    ----------
    >>> from skfp.distances import dice_binary_distance
    >>> import numpy as np
    >>> vec_a = np.array([1, 0, 1])
    >>> vec_b = np.array([1, 0, 1])
    >>> dist = dice_binary_distance(vec_a, vec_b)
    >>> dist
    0.0

    >>> from skfp.distances import dice_binary_distance
    >>> from scipy.sparse import csr_array
    >>> vec_a = csr_array([[1, 0, 1]])
    >>> vec_b = csr_array([[1, 0, 1]])
    >>> dist = dice_binary_distance(vec_a, vec_b)
    >>> dist
    0.0
    """

    return 1 - dice_binary_similarity(vec_a, vec_b)
