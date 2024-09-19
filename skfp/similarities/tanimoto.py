from typing import Union

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils._param_validation import validate_params


def binary_tanimoto_similarity(
    A: Union[np.ndarray, csr_matrix], B: Union[np.ndarray, csr_matrix]
) -> float:
    """
    Computes the Tanimoto similarity [1]_ for binary data between two input arrays
    or sprase matrices. Calls suitable private method based on input type.

    Parameters
    ----------
    A : np.darray or scipy.sparse.csr_matrix
        First binary input array or sparse matrix.
    B : np.darray or scipy.sparse.csr_matrix
        Second binary input array or sparse matrix.

    Returns
    ----------
    tanimoto_similarity : float
        Tanimoto similarity between A and B.

    References
    ----------
    .. [1] `Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3`_
    """
    if isinstance(A, csr_matrix) and isinstance(B, csr_matrix):
        return _binary_scipy_tanimoto(A, B)
    elif isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        return _binary_numpy_tanimoto(A, B)
    else:
        raise TypeError(
            f"Both A and B must be of the same type: either numpy.ndarray "
            f"or scipy.sparse.csr_matrix, got {type(A)} and {type(B)}"
        )


def binary_tanimoto_distance(
    A: Union[np.ndarray, csr_matrix], B: Union[np.ndarray, csr_matrix]
) -> float:
    """
    Computes the Tanimoto distance for binary data between two input arrays or sparse matrices.

    Parameters
    ----------
    A : np.darray or scipy.sparse.csr_matrix
        First binary input array or sparse matrix.
    B : np.darray or scipy.sparse.csr_matrix
        Second binary input array or sparse matrix.

    Returns
    ----------
    float
        Tanimoto distance between A and B.
    """

    return 1 - binary_tanimoto_similarity(A, B)


def count_tanimoto_similarity(
    A: Union[np.ndarray, csr_matrix], B: Union[np.ndarray, csr_matrix]
) -> float:
    """
    Computes the Tanimoto similarity [1]_ for continuous data between two input arrays
    or sprase matrices. Calls suitable private method based on input type.

    Parameters
    ----------
    A : np.darray or scipy.sparse.csr_matrix
        First binary input array or sparse matrix.
    B : np.darray or scipy.sparse.csr_matrix
        Second binary input array or sparse matrix.

    Returns
    ----------
    tanimoto_similarity : float
        Tanimoto similarity between A and B.

    References
    ----------
    .. [1] `Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3`_
    """
    if isinstance(A, csr_matrix) and isinstance(B, csr_matrix):
        return _count_scipy_tanimoto(A, B)
    elif isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        return _count_numpy_tanimoto(A, B)
    else:
        raise TypeError(
            f"Both A and B must be of the same type: either numpy.ndarray "
            f"or scipy.sparse.csr_matrix, got {type(A)} and {type(B)}"
        )


def count_tanimoto_distance(
    A: Union[np.ndarray, csr_matrix], B: Union[np.ndarray, csr_matrix]
) -> float:
    """
    Computes the Tanimoto distance for binary data between two input arrays or sparse matrices.

    Parameters
    ----------
    A : np.darray or scipy.sparse.csr_matrix
        First binary input array or sparse matrix.
    B : np.darray or scipy.sparse.csr_matrix
        Second binary input array or sparse matrix.

    Returns
    ----------
    float
        Tanimoto distance between A and B.
    """
    return 1 - count_tanimoto_similarity(A, B)


def _binary_numpy_tanimoto(A: np.ndarray, B: np.ndarray) -> float:
    """
    Calculate the Tanimoto similarity between two binary numpy arrays.
    """
    _check_nan(A)
    _check_nan(B)

    if np.sum(A) == 0 and np.sum(B) == 0:
        return 1.0

    dot_product_AB: float = np.dot(A, B)
    dot_product_AA: float = np.sum(A)
    dot_product_BB: float = np.sum(B)

    denominator: float = dot_product_AA + dot_product_BB - dot_product_AB
    _check_zero_denominator(denominator)

    tanimoto_similarity = dot_product_AB / denominator
    return tanimoto_similarity


def _binary_scipy_tanimoto(A: csr_matrix, B: csr_matrix) -> float:
    """
    Calculate the Tanimoto similarity between two binary scipy arrays.
    """
    _check_nan(A)
    _check_nan(B)

    if np.sum(A) == 0 and np.sum(B) == 0:
        return 1.0

    dot_product_AB: float = A.multiply(B).sum()
    dot_product_AA: float = A.sum()
    dot_product_BB: float = B.sum()

    denominator: float = dot_product_AA + dot_product_BB - dot_product_AB
    _check_zero_denominator(denominator)

    tanimoto_coeff: float = dot_product_AB / denominator

    return tanimoto_coeff


def _count_numpy_tanimoto(A: np.ndarray, B: np.ndarray) -> float:
    """
    Calculate the Tanimoto similarity between two continuous numpy arrays.
    """
    _check_nan(A)
    _check_nan(B)

    if np.sum(A) == 0 and np.sum(B) == 0:
        return 1.0

    dot_product_AB: float = np.dot(A, B)
    dot_product_AA: float = np.dot(A, A)
    dot_product_BB: float = np.dot(B, B)

    denominator: float = dot_product_AA + dot_product_BB - dot_product_AB
    _check_zero_denominator(denominator)

    tanimoto_coeff: float = dot_product_AB / denominator

    return tanimoto_coeff


def _count_scipy_tanimoto(A: csr_matrix, B: csr_matrix) -> float:
    """
    Calculate the Tanimoto similarity between two continuous scipy arrays.
    """
    _check_nan(A)
    _check_nan(B)

    if np.sum(A) == 0 and np.sum(B) == 0:
        return 1.0

    dot_product_AB: float = A.multiply(B).sum()
    dot_product_AA: float = A.multiply(A).sum()
    dot_product_BB: float = B.multiply(B).sum()

    denominator: float = dot_product_AA + dot_product_BB - dot_product_AB
    _check_zero_denominator(denominator)

    tanimoto_coeff: float = dot_product_AB / denominator

    return tanimoto_coeff


def _check_nan(arr: Union[np.ndarray, csr_matrix]) -> None:
    """
    Check if passed numpy array or scipy sparse matrix contains NaN values.
    """
    if isinstance(arr, np.ndarray):
        if np.isnan(arr).any():
            raise ValueError("Input array contains NaN values")
    elif isinstance(arr, csr_matrix):
        if np.isnan(arr.data).any():
            raise ValueError("Input sparse matrix contains NaN values")
    else:
        raise TypeError(
            "Unsupported type provided. Expected numpy.ndarray or scipy.sparse.csr_matrix."
        )


def _check_zero_denominator(denominator: float) -> None:
    """
    Check if passed expression is 0.0 to prevent zero division while calculating Tanimoto.
    """
    if denominator == 0.0:
        raise ZeroDivisionError("Denominator is zero")
