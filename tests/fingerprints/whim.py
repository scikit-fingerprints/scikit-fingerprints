import numpy as np
from rdkit.Chem.rdMolDescriptors import CalcWHIM
from scipy.sparse import csr_array

from skfp.fingerprints import WHIMFingerprint


def test_whim_fingerprint(mols_conformers_list):
    whim_fp = WHIMFingerprint(n_jobs=-1)
    X_skfp = whim_fp.transform(mols_conformers_list)

    X_rdkit = np.array(
        [
            CalcWHIM(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_list
        ]
    )
    X_rdkit = np.minimum(X_rdkit, whim_fp.clip_val)

    assert np.allclose(X_skfp, X_rdkit, atol=1e-1)
    assert X_skfp.shape == (len(mols_conformers_list), 114)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_whim_sparse_fingerprint(mols_conformers_list):
    whim_fp = WHIMFingerprint(sparse=True, n_jobs=-1)
    X_skfp = whim_fp.transform(mols_conformers_list)

    X_rdkit = csr_array(
        [
            CalcWHIM(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_list
        ]
    )
    X_rdkit = X_rdkit.minimum(whim_fp.clip_val)

    assert np.allclose(X_skfp.data, X_rdkit.data, atol=1e-1)
    assert X_skfp.shape == (len(mols_conformers_list), 114)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_whim_feature_names():
    whim_fp = WHIMFingerprint()
    feature_names = whim_fp.get_feature_names_out()

    assert len(feature_names) == whim_fp.n_features_out
    assert len(feature_names) == len(set(feature_names))

    assert feature_names[0] == "unweighted axis 1 directional WHIM size"
    assert feature_names[3] == "unweighted axis 1 directional WHIM shape"
    assert feature_names[10] == "unweighted axis 3 directional WHIM density"
    assert feature_names[11] == "atomic mass axis 1 directional WHIM size"
    assert feature_names[22] == "van der Waals volume axis 1 directional WHIM size"
    assert feature_names[33] == "electronegativity axis 1 directional WHIM size"
    assert feature_names[44] == "polarizability axis 1 directional WHIM size"
    assert feature_names[76] == "IState axis 3 directional WHIM density"

    assert feature_names[77] == "unweighted global WHIM size"
    assert feature_names[78] == "atomic mass global WHIM size"
    assert feature_names[83] == "IState global WHIM size"

    assert all(
        name.endswith("global WHIM size cross sums") for name in feature_names[84:91]
    )
    assert feature_names[84].startswith("unweighted")
    assert feature_names[90].startswith("IState")

    assert feature_names[91] == "unweighted global WHIM axial shape"
    assert feature_names[92] == "atomic mass global WHIM axial shape"

    assert all(name.endswith("global WHIM shape") for name in feature_names[93:100])
    assert feature_names[93].startswith("unweighted")
    assert feature_names[99].startswith("IState")

    assert all(name.endswith("global WHIM density") for name in feature_names[100:107])
    assert feature_names[100].startswith("unweighted")
    assert feature_names[106].startswith("IState")

    assert all(name.endswith("global WHIM symmetry") for name in feature_names[107:])
    assert feature_names[107].startswith("unweighted")
    assert feature_names[113].startswith("IState")
