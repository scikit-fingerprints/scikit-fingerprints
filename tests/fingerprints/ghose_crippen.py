import numpy as np
import pytest
from scipy.sparse import csr_array
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import GhoseCrippenFingerprint


def test_ghose_crippen_bit_fingerprint(smiles_list):
    fp = GhoseCrippenFingerprint(sparse=False, count=False, n_jobs=-1)
    X = fp.transform(smiles_list)

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.uint8
    assert X.shape == (len(smiles_list), 110)
    assert np.all(np.isin(X, [0, 1]))


def test_ghose_crippen_count_fingerprint(smiles_list):
    fp = GhoseCrippenFingerprint(sparse=False, count=True, n_jobs=-1)
    X = fp.transform(smiles_list)

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.uint32
    assert X.shape == (len(smiles_list), 110)
    assert np.all(X >= 0)


def test_ghose_crippen_bit_sparse_fingerprint(smiles_list):
    fp = GhoseCrippenFingerprint(sparse=True, count=False, n_jobs=-1)
    X = fp.transform(smiles_list)

    assert isinstance(X, csr_array)
    assert X.dtype == np.uint8
    assert X.shape == (len(smiles_list), 110)
    assert np.all(X.data == 1)


def test_ghose_crippen_count_sparse_fingerprint(smiles_list):
    fp = GhoseCrippenFingerprint(sparse=True, count=True, n_jobs=-1)
    X = fp.transform(smiles_list)

    assert isinstance(X, csr_array)
    assert X.dtype == np.uint32
    assert X.shape == (len(smiles_list), 110)
    assert np.all(X.data > 0)
