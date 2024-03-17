import numpy as np
from scipy.sparse import csr_array

from skfp.fingerprints import BTFingerprint


def test_bt_bit_fingerprint(smiles_list):
    bt_fp = BTFingerprint(sparse=False, count=False, n_jobs=-1)
    X_skfp = bt_fp.transform(smiles_list)

    assert isinstance(X_skfp, np.ndarray)
    assert X_skfp.shape == (len(smiles_list), bt_fp.fp_size)
    assert np.all(np.isin(X_skfp, [0, 1]))
    assert np.issubdtype(X_skfp.dtype, int)


def test_bt_count_fingerprint(smiles_list):
    bt_fp = BTFingerprint(sparse=False, count=True, n_jobs=-1)
    X_skfp = bt_fp.transform(smiles_list)

    assert isinstance(X_skfp, np.ndarray)
    assert X_skfp.shape == (len(smiles_list), bt_fp.fp_size)
    assert np.all(X_skfp >= 0)
    assert np.issubdtype(X_skfp.dtype, int)


def test_bt_sparse_bit_fingerprint(smiles_list):
    bt_fp = BTFingerprint(sparse=True, count=False, n_jobs=-1)
    X_skfp = bt_fp.transform(smiles_list)

    assert isinstance(X_skfp, csr_array)
    assert X_skfp.shape == (len(smiles_list), bt_fp.fp_size)
    assert np.all(np.isin(X_skfp.data, [0, 1]))
    assert np.issubdtype(X_skfp.dtype, int)


def test_bt_sparse_count_fingerprint(smiles_list):
    bt_fp = BTFingerprint(sparse=True, count=True, n_jobs=-1)
    X_skfp = bt_fp.transform(smiles_list)

    assert isinstance(X_skfp, csr_array)
    assert X_skfp.shape == (len(smiles_list), bt_fp.fp_size)
    assert np.all(X_skfp.data >= 0)
    assert np.issubdtype(X_skfp.dtype, int)
