import numpy as np
from scipy.sparse import csr_array

from skfp.fingerprints import MQNsFingerprint


def test_mqns_bit_fingerprint(smiles_list):
    fp = MQNsFingerprint(count=False, n_jobs=-1)
    X = fp.transform(smiles_list)

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.uint8
    assert X.shape == (len(smiles_list), 42)
    assert np.all(np.isin(X, [0, 1]))


def test_mqns_count_fingerprint(smiles_list):
    fp = MQNsFingerprint(count=True, n_jobs=-1)
    X = fp.transform(smiles_list)

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.uint32
    assert X.shape == (len(smiles_list), 42)
    assert np.all(X >= 0)


def test_mqns_bit_sparse_fingerprint(smiles_list):
    fp = MQNsFingerprint(count=False, sparse=True, n_jobs=-1)
    X = fp.transform(smiles_list)

    assert isinstance(X, csr_array)
    assert X.dtype == np.uint8
    assert X.shape == (len(smiles_list), 42)
    assert np.all(X.data == 1)


def test_mqns_count_sparse_fingerprint(smiles_list):
    fp = MQNsFingerprint(count=True, sparse=True, n_jobs=-1)
    X = fp.transform(smiles_list)

    assert isinstance(X, csr_array)
    assert X.dtype == np.uint32
    assert X.shape == (len(smiles_list), 42)
    assert np.all(X.data > 0)
