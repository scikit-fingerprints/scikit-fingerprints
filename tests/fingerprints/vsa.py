import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from rdkit.Chem.EState.EState_VSA import EState_VSA_
from rdkit.Chem.rdMolDescriptors import PEOE_VSA_, SMR_VSA_, SlogP_VSA_
from scipy.sparse import csr_array

from skfp.fingerprints import VSAFingerprint


def test_vsa_fingerprint(mols_list):
    vsa_fp = VSAFingerprint(n_jobs=-1)
    X_skfp = vsa_fp.transform(mols_list)

    X_slogp = np.array([SlogP_VSA_(mol) for mol in mols_list])
    X_smr = np.array([SMR_VSA_(mol) for mol in mols_list])
    X_peoe = np.array([PEOE_VSA_(mol) for mol in mols_list])

    X_rdkit = np.column_stack((X_slogp, X_smr, X_peoe))

    assert_allclose(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(mols_list), 36))
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_vsa_fingerprint_with_estate(mols_list):
    vsa_fp = VSAFingerprint(variant="all", n_jobs=-1)
    X_skfp = vsa_fp.transform(mols_list)

    X_slogp = np.array([SlogP_VSA_(mol) for mol in mols_list])
    X_smr = np.array([SMR_VSA_(mol) for mol in mols_list])
    X_peoe = np.array([PEOE_VSA_(mol) for mol in mols_list])
    X_estate = np.array([EState_VSA_(mol) for mol in mols_list])

    X_rdkit = np.column_stack((X_slogp, X_smr, X_peoe, X_estate))

    assert_allclose(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(mols_list), 47))
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_vsa_fingerprint_variants(mols_list):
    for variant, expected_num_cols in [
        ("SlogP", 12),
        ("SMR", 10),
        ("PEOE", 14),
        ("EState", 11),
        ("all_original", 36),
        ("all", 47),
    ]:
        vsa_fp = VSAFingerprint(variant=variant, n_jobs=-1)
        X_skfp = vsa_fp.transform(mols_list)

        assert_equal(X_skfp.shape, (len(mols_list), expected_num_cols))
        assert np.issubdtype(X_skfp.dtype, np.floating)


def test_vsa_sparse_fingerprint(mols_list):
    vsa_fp = VSAFingerprint(sparse=True, n_jobs=-1)
    X_skfp = vsa_fp.transform(mols_list)

    X_slogp = np.array([SlogP_VSA_(mol) for mol in mols_list])
    X_smr = np.array([SMR_VSA_(mol) for mol in mols_list])
    X_peoe = np.array([PEOE_VSA_(mol) for mol in mols_list])

    X_rdkit = csr_array(np.column_stack((X_slogp, X_smr, X_peoe)))

    assert_allclose(X_skfp.data, X_rdkit.data)
    assert_equal(X_skfp.shape, (len(mols_list), 36))
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_vsa_slogp_feature_names():
    vsa_fp = VSAFingerprint(variant="SlogP", n_jobs=-1)
    feature_names = vsa_fp.get_feature_names_out()

    assert_equal(len(feature_names), vsa_fp.n_features_out)
    assert_equal(len(feature_names), len(set(feature_names)))

    assert_equal(feature_names[0], "SlogP < -0.4")
    assert_equal(feature_names[1], "-0.4 <= SlogP < -0.2")
    assert_equal(feature_names[-2], "0.5 <= SlogP < 0.6")
    assert_equal(feature_names[-1], "SlogP >= 0.6")


def test_vsa_smr_feature_names():
    vsa_fp = VSAFingerprint(variant="SMR", n_jobs=-1)
    feature_names = vsa_fp.get_feature_names_out()

    assert_equal(len(feature_names), vsa_fp.n_features_out)
    assert_equal(len(feature_names), len(set(feature_names)))

    assert_equal(feature_names[0], "SMR < 1.29")
    assert_equal(feature_names[1], "1.29 <= SMR < 1.82")
    assert_equal(feature_names[-2], "3.8 <= SMR < 4.0")
    assert_equal(feature_names[-1], "SMR >= 4.0")


def test_vsa_peoe_feature_names():
    vsa_fp = VSAFingerprint(variant="PEOE", n_jobs=-1)
    feature_names = vsa_fp.get_feature_names_out()

    assert_equal(len(feature_names), vsa_fp.n_features_out)
    assert_equal(len(feature_names), len(set(feature_names)))

    assert_equal(feature_names[0], "PEOE < -0.3")
    assert_equal(feature_names[1], "-0.3 <= PEOE < -0.25")
    assert_equal(feature_names[-2], "0.25 <= PEOE < 0.3")
    assert_equal(feature_names[-1], "PEOE >= 0.3")


def test_vsa_estate_feature_names():
    vsa_fp = VSAFingerprint(variant="EState", n_jobs=-1)
    feature_names = vsa_fp.get_feature_names_out()

    assert_equal(len(feature_names), vsa_fp.n_features_out)
    assert_equal(len(feature_names), len(set(feature_names)))

    assert_equal(feature_names[0], "EState < -0.39")
    assert_equal(feature_names[1], "-0.39 <= EState < 0.29")
    assert_equal(feature_names[-2], "9.17 <= EState < 15.0")
    assert_equal(feature_names[-1], "EState >= 15.0")


def test_vsa_all_feature_names():
    vsa_fp = VSAFingerprint(variant="all", n_jobs=-1)
    feature_names = vsa_fp.get_feature_names_out()

    assert_equal(len(feature_names), vsa_fp.n_features_out)
    assert_equal(len(feature_names), len(set(feature_names)))

    assert all("SlogP" in name for name in feature_names[0:12])
    assert all("SMR" in name for name in feature_names[12:22])
    assert all("PEOE" in name for name in feature_names[22:36])
    assert all("EState" in name for name in feature_names[36:47])


def test_vsa_variants():
    # proper variants, should just work
    for variant in ["SlogP", "SMR", "PEOE", "EState", "all_original", "all"]:
        VSAFingerprint(variant=variant)

    with pytest.raises(ValueError, match='Variant "nonexistent" not recognized'):
        VSAFingerprint(variant="nonexistent")
