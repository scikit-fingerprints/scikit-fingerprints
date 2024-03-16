from fingerprints import GETAWAYFingerprint


def test_getaway_bit_fingerprint(mols_conformers_list):
    getaway_fp = GETAWAYFingerprint(sparse=False, n_jobs=-1)
    X_skfp = getaway_fp.transform(mols_conformers_list)

    assert X_skfp.shape == (len(mols_conformers_list), 273)


def test_getaway_sparse_bit_fingerprint(mols_conformers_list):
    getaway_fp = GETAWAYFingerprint(sparse=True, n_jobs=-1)
    X_skfp = getaway_fp.transform(mols_conformers_list)

    assert X_skfp.shape == (len(mols_conformers_list), 273)
