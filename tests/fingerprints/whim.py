import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from rdkit.Chem.rdMolDescriptors import CalcWHIM
from scipy.sparse import csr_array

from skfp.fingerprints import WHIMFingerprint


@pytest.fixture
def mols_conformers_3_plus_atoms(mols_conformers_list):
    return [mol for mol in mols_conformers_list if mol.GetNumAtoms() >= 3]


def test_whim_fingerprint(mols_conformers_3_plus_atoms):
    whim_fp = WHIMFingerprint(n_jobs=-1)
    X_skfp = whim_fp.transform(mols_conformers_3_plus_atoms)

    X_rdkit = np.array(
        [
            CalcWHIM(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_3_plus_atoms
        ]
    )
    X_rdkit = np.clip(X_rdkit, -whim_fp.clip_val, whim_fp.clip_val)

    for i in range(X_skfp.shape[0]):
        x_skfp = X_skfp[i]
        x_rdkit = X_rdkit[i]
        for j in range(len(x_skfp)):
            if abs(x_skfp[j] - x_rdkit[j]) > 100:
                print(whim_fp.get_feature_names_out()[j], x_skfp[j], x_rdkit[j])

    assert_allclose(X_skfp, X_rdkit, atol=1e-1)
    assert_equal(X_skfp.shape, (len(mols_conformers_3_plus_atoms), 114))
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_whim_sparse_fingerprint(mols_conformers_3_plus_atoms):
    whim_fp = WHIMFingerprint(sparse=True, n_jobs=-1)
    X_skfp = whim_fp.transform(mols_conformers_3_plus_atoms)

    X_rdkit = csr_array(
        [
            CalcWHIM(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_3_plus_atoms
        ]
    )
    X_rdkit = X_rdkit.minimum(whim_fp.clip_val)

    assert_allclose(X_skfp.data, X_rdkit.data, atol=1e-1)
    assert_equal(X_skfp.shape, (len(mols_conformers_3_plus_atoms), 114))
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_whim_feature_names():
    whim_fp = WHIMFingerprint()
    feature_names = whim_fp.get_feature_names_out()

    assert_equal(len(feature_names), whim_fp.n_features_out)
    assert_equal(len(feature_names), len(set(feature_names)))

    assert_equal(feature_names[0], "unweighted axis 1 directional WHIM size")
    assert_equal(feature_names[3], "unweighted axis 1 directional WHIM shape")
    assert_equal(feature_names[10], "unweighted axis 3 directional WHIM density")
    assert_equal(feature_names[11], "atomic mass axis 1 directional WHIM size")
    assert_equal(feature_names[22], "van der Waals volume axis 1 directional WHIM size")
    assert_equal(feature_names[33], "electronegativity axis 1 directional WHIM size")
    assert_equal(feature_names[44], "polarizability axis 1 directional WHIM size")
    assert_equal(feature_names[76], "IState axis 3 directional WHIM density")

    assert_equal(feature_names[77], "unweighted global WHIM size")
    assert_equal(feature_names[78], "atomic mass global WHIM size")
    assert_equal(feature_names[83], "IState global WHIM size")

    assert all(
        name.endswith("global WHIM size cross sums") for name in feature_names[84:91]
    )
    assert feature_names[84].startswith("unweighted")
    assert feature_names[90].startswith("IState")

    assert_equal(feature_names[91], "unweighted global WHIM axial shape")
    assert_equal(feature_names[92], "atomic mass global WHIM axial shape")

    assert all(name.endswith("global WHIM shape") for name in feature_names[93:100])
    assert feature_names[93].startswith("unweighted")
    assert feature_names[99].startswith("IState")

    assert all(name.endswith("global WHIM density") for name in feature_names[100:107])
    assert feature_names[100].startswith("unweighted")
    assert feature_names[106].startswith("IState")

    assert all(name.endswith("global WHIM symmetry") for name in feature_names[107:])
    assert feature_names[107].startswith("unweighted")
    assert feature_names[113].startswith("IState")
