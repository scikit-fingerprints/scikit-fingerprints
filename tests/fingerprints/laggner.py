import numpy as np
from scipy.sparse import csr_array

from skfp.fingerprints import LaggnerFingerprint


def test_laggner_bit_fingerprint(smiles_list):
    fp = LaggnerFingerprint(n_jobs=-1)
    X = fp.transform(smiles_list)

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.uint8
    assert X.shape == (len(smiles_list), 307)
    assert np.all(np.isin(X, [0, 1]))


def test_laggner_count_fingerprint(smiles_list):
    fp = LaggnerFingerprint(count=True, n_jobs=-1)
    X = fp.transform(smiles_list)

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.uint32
    assert X.shape == (len(smiles_list), 307)
    assert np.all(X >= 0)


def test_laggner_bit_sparse_fingerprint(smiles_list):
    fp = LaggnerFingerprint(sparse=True, n_jobs=-1)
    X = fp.transform(smiles_list)

    assert isinstance(X, csr_array)
    assert X.dtype == np.uint8
    assert X.shape == (len(smiles_list), 307)
    assert np.all(X.data == 1)


def test_laggner_count_sparse_fingerprint(smiles_list):
    fp = LaggnerFingerprint(sparse=True, count=True, n_jobs=-1)
    X = fp.transform(smiles_list)

    assert isinstance(X, csr_array)
    assert X.dtype == np.uint32
    assert X.shape == (len(smiles_list), 307)
    assert np.all(X.data > 0)


def test_laggner_salt_matching():
    non_salts = ["O", "CC", "[C-]#N", "CC=O"]
    salts = ["[Na+].[Cl-]", "[Ca+2].[Cl-].[Cl-]"]
    smiles_list = non_salts + salts

    fp = LaggnerFingerprint()
    X = fp.transform(smiles_list)

    assert X[0, 298] == 0
    assert X[1, 298] == 0
    assert X[2, 298] == 0
    assert X[3, 298] == 0
    assert X[4, 298] == 1
    assert X[5, 298] == 1
