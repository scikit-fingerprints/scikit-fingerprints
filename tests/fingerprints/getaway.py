import numpy as np

from skfp.fingerprints import GETAWAYFingerprint


def test_getaway_fingerprint(mols_conformers_list):
    getaway_fp = GETAWAYFingerprint(n_jobs=-1)
    X_skfp = getaway_fp.transform(mols_conformers_list)

    assert X_skfp.shape == (len(mols_conformers_list), 273)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_getaway_sparse_fingerprint(mols_conformers_list):
    getaway_fp = GETAWAYFingerprint(sparse=True, n_jobs=-1)
    X_skfp = getaway_fp.transform(mols_conformers_list)

    assert X_skfp.shape == (len(mols_conformers_list), 273)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_getaway_feature_names():
    getaway_fp = GETAWAYFingerprint()
    feature_names = getaway_fp.get_feature_names_out()

    assert len(feature_names) == getaway_fp.n_features_out
    assert len(feature_names) == len(set(feature_names))

    # for indices understanding, see:
    # https://github.com/rdkit/rdkit/blob/53a01430e057d34cf7a7ca52eb2d8b069178114d/Code/GraphMol/Descriptors/GETAWAY.cpp#L1145
    # and slide 28 from https://github.com/rdkit/UGM_2017/blob/master/Presentations/Godin_3D_Descriptors.pdf
    # note that here we index from zero

    assert "ITH" in feature_names[0]
    assert "ISH" in feature_names[1]
    assert "HIC" in feature_names[2]
    assert "HGM" in feature_names[3]

    assert all(
        "total H index" in feature_names[idx] for idx in [13, 33, 53, 73, 93, 113, 133]
    )
    assert all(
        "total HATS index" in feature_names[idx]
        for idx in [23, 43, 63, 83, 103, 123, 143]
    )

    assert "RCON" in feature_names[144]
    assert "RARS" in feature_names[145]
    assert "REIG" in feature_names[146]

    assert all(
        "R total index" in feature_names[idx]
        for idx in [155, 173, 191, 209, 227, 254, 263]
    )
    assert all(
        "maximal R total index" in feature_names[idx]
        for idx in [164, 182, 200, 218, 236, 254, 272]
    )
