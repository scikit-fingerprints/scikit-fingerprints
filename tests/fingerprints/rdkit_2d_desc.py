import numpy as np

from skfp.fingerprints import RDKit2DDescriptorsFingerprint


def test_rdkit_2d_desc_fingerprint(smallest_mols_list):
    getaway_fp = RDKit2DDescriptorsFingerprint(sparse=False, n_jobs=-1)
    X_skfp = getaway_fp.transform(smallest_mols_list)

    assert X_skfp.shape == (len(smallest_mols_list), 200)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_rdkit_2d_desc_sparse_fingerprint(smallest_mols_list):
    getaway_fp = RDKit2DDescriptorsFingerprint(sparse=True, n_jobs=-1)
    X_skfp = getaway_fp.transform(smallest_mols_list)

    assert X_skfp.shape == (len(smallest_mols_list), 200)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_rdkit_2d_desc_normalized_fingerprint(smallest_mols_list):
    getaway_fp = RDKit2DDescriptorsFingerprint(normalized=True, sparse=False, n_jobs=-1)
    X_skfp = getaway_fp.transform(smallest_mols_list)

    assert X_skfp.shape == (len(smallest_mols_list), 200)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_rdkit_2d_desc_normalized_sparse_fingerprint(smallest_mols_list):
    getaway_fp = RDKit2DDescriptorsFingerprint(normalized=True, sparse=False, n_jobs=-1)
    X_skfp = getaway_fp.transform(smallest_mols_list)

    assert X_skfp.shape == (len(smallest_mols_list), 200)
    assert np.issubdtype(X_skfp.dtype, np.floating)
