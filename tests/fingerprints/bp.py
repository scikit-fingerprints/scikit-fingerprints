import numpy as np
from scipy.sparse import csr_array

from skfp.fingerprints import BPFingerprint


def test_bp_bit_fingerprint(smiles_list):
    bp_fp = BPFingerprint(sparse=False, count=False, n_jobs=-1)
    X_skfp = bp_fp.transform(smiles_list)

    assert isinstance(X_skfp, np.ndarray)
    assert X_skfp.shape == (len(smiles_list), bp_fp.fp_size)
    assert np.all(np.isin(X_skfp, [0, 1]))
    assert np.issubdtype(X_skfp.dtype, int)


def test_bp_count_fingerprint(smiles_list):
    bp_fp = BPFingerprint(sparse=False, count=True, n_jobs=-1)
    X_skfp = bp_fp.transform(smiles_list)

    assert isinstance(X_skfp, np.ndarray)
    assert X_skfp.shape == (len(smiles_list), bp_fp.fp_size)
    assert np.all(X_skfp >= 0)
    assert np.issubdtype(X_skfp.dtype, int)


def test_bp_sparse_bit_fingerprint(smiles_list):
    bp_fp = BPFingerprint(sparse=True, count=False, n_jobs=-1)
    X_skfp = bp_fp.transform(smiles_list)

    assert isinstance(X_skfp, csr_array)
    assert X_skfp.shape == (len(smiles_list), bp_fp.fp_size)
    assert np.all(np.isin(X_skfp.data, [0, 1]))
    assert np.issubdtype(X_skfp.dtype, int)


def test_bp_sparse_count_fingerprint(smiles_list):
    bp_fp = BPFingerprint(sparse=True, count=True, n_jobs=-1)
    X_skfp = bp_fp.transform(smiles_list)

    assert isinstance(X_skfp, csr_array)
    assert X_skfp.shape == (len(smiles_list), bp_fp.fp_size)
    assert np.all(X_skfp.data >= 0)
    assert np.issubdtype(X_skfp.dtype, int)
