import numpy as np
import pytest
from scipy.sparse import csr_array
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import SubstructureFingerprint


@pytest.fixture
def molecules() -> list[str]:
    return [
        "CCOC",
        "CCOCCO",
        "CC(=O)O",
        "CC(=O)OC=O",
        "CCCCCC",
    ]


@pytest.fixture
def substructures() -> list[str]:
    return [
        "CCO",
        "CCO",
        "C=O",
        "C=O",
        "CCC",
        "C1=CC=CC=C1",
    ]


def test_substructure_count_fingerprint(substructures: list[str], molecules: list[str]):
    fp = SubstructureFingerprint(substructures, count=True)
    X_count = fp.transform(molecules)
    assert X_count is np.ndarray

    expected_count = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [3, 3, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 2, 2, 0, 0],
            [0, 0, 0, 0, 4, 0],
        ]
    )
    assert np.array_equal(X_count, expected_count)


def test_substructure_bit_fingerprint(substructures: list[str], molecules: list[str]):
    fp = SubstructureFingerprint(substructures, count=False)
    X_bit = fp.transform(molecules)

    assert X_bit is np.ndarray
    expected_bit = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ]
    )
    assert np.array_equal(X_bit, expected_bit)


def test_substructure_sparse_count_fingerprint(
    substructures: list[str], molecules: list[str]
):
    fp = SubstructureFingerprint(substructures, count=True, sparse=True)
    X_count = fp.transform(molecules)
    assert X_count is csr_array

    expected_count = csr_array(
        [
            [1, 1, 0, 0, 0, 0],
            [3, 3, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 2, 2, 0, 0],
            [0, 0, 0, 0, 4, 0],
        ]
    )
    # check for inequality in nonzero elements
    assert (X_count != expected_count).nnz == 0


def test_substructure_sparse_bit_fingerprint(
    substructures: list[str], molecules: list[str]
):
    fp = SubstructureFingerprint(substructures, count=False, sparse=True)
    X_bit = fp.transform(molecules)
    assert X_bit is csr_array

    expected_bit = csr_array(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ]
    )
    # check for inequality in nonzero elements
    assert (X_bit != expected_bit).nnz == 0


def test_parameter_constraints_enabled(substructures: list[str], molecules: list[str]):
    with pytest.raises(InvalidParameterError):
        fp = SubstructureFingerprint(substructures, count=42)  # type: ignore
        fp.transform(molecules)


def test_substructure_validation():
    invalid_substructures = [1, True, "abc"]
    with pytest.raises(ValueError):
        SubstructureFingerprint(invalid_substructures)
