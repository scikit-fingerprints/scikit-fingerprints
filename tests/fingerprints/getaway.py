from fingerprints import GETAWAYFingerprint
from preprocessing import ConformerGenerator


def test_getaway_bit_fingerprint(smallest_mols_list):
    conf_gen = ConformerGenerator()
    mols = conf_gen.transform(smallest_mols_list)

    getaway_fp = GETAWAYFingerprint(sparse=False, n_jobs=1)
    X_skfp = getaway_fp.transform(mols)

    assert X_skfp.shape == (len(mols), 273)


def test_getaway_sparse_bit_fingerprint(smallest_mols_list):
    conf_gen = ConformerGenerator()
    mols = conf_gen.transform(smallest_mols_list)

    getaway_fp = GETAWAYFingerprint(sparse=True, n_jobs=-1)
    X_skfp = getaway_fp.transform(mols)

    assert X_skfp.shape == (len(mols), 273)
