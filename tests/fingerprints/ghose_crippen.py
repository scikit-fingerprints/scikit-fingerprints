import numpy as np
from scipy.sparse import csr_array

from skfp.fingerprints import GhoseCrippenFingerprint


def test_ghose_crippen_bit_fingerprint(smiles_list):
    gc_fp = GhoseCrippenFingerprint(n_jobs=-1)
    X = gc_fp.transform(smiles_list)

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.uint8
    assert X.shape == (len(smiles_list), 110)
    assert np.all(np.isin(X, [0, 1]))


def test_ghose_crippen_count_fingerprint(smiles_list):
    gc_fp = GhoseCrippenFingerprint(count=True, n_jobs=-1)
    X = gc_fp.transform(smiles_list)

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.uint32
    assert X.shape == (len(smiles_list), 110)
    assert np.all(X >= 0)


def test_ghose_crippen_bit_sparse_fingerprint(smiles_list):
    gc_fp = GhoseCrippenFingerprint(sparse=True, n_jobs=-1)
    X = gc_fp.transform(smiles_list)

    assert isinstance(X, csr_array)
    assert X.dtype == np.uint8
    assert X.shape == (len(smiles_list), 110)
    assert np.all(X.data == 1)


def test_ghose_crippen_count_sparse_fingerprint(smiles_list):
    gc_fp = GhoseCrippenFingerprint(sparse=True, count=True, n_jobs=-1)
    X = gc_fp.transform(smiles_list)

    assert isinstance(X, csr_array)
    assert X.dtype == np.uint32
    assert X.shape == (len(smiles_list), 110)
    assert np.all(X.data > 0)


def test_ghose_crippen_feature_names():
    gc_fp = GhoseCrippenFingerprint()
    feature_names = gc_fp.get_feature_names_out()
    assert len(feature_names) == gc_fp.n_features_out
