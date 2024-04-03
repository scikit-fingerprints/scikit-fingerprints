import numpy as np

from skfp.fingerprints import GETAWAYFingerprint


def test_getaway_fingerprint(mols_conformers_list):
    getaway_fp = GETAWAYFingerprint(sparse=False, n_jobs=-1)
    X_skfp = getaway_fp.transform(mols_conformers_list)

    assert X_skfp.shape == (len(mols_conformers_list), 273)
    assert np.issubdtype(X_skfp, np.floating)


def test_getaway_sparse_fingerprint(mols_conformers_list):
    getaway_fp = GETAWAYFingerprint(sparse=True, n_jobs=-1)
    X_skfp = getaway_fp.transform(mols_conformers_list)

    assert X_skfp.shape == (len(mols_conformers_list), 273)
    assert np.issubdtype(X_skfp.dtype, np.floating)
