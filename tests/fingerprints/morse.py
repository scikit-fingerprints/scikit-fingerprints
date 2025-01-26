import numpy as np
from rdkit.Chem.rdMolDescriptors import CalcMORSE
from scipy.sparse import csr_array

from skfp.fingerprints import MORSEFingerprint


def test_morse_fingerprint(mols_conformers_list):
    morse_fp = MORSEFingerprint(n_jobs=-1)
    X_skfp = morse_fp.transform(mols_conformers_list)

    X_rdkit = np.array(
        [
            CalcMORSE(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_list
        ]
    )

    assert np.allclose(X_skfp, X_rdkit, atol=1e-1)
    assert X_skfp.shape == (len(mols_conformers_list), 224)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_morse_sparse_fingerprint(mols_conformers_list):
    morse_fp = MORSEFingerprint(sparse=True, n_jobs=-1)
    X_skfp = morse_fp.transform(mols_conformers_list)

    X_rdkit = csr_array(
        [
            CalcMORSE(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_list
        ]
    )

    assert np.allclose(X_skfp.data, X_rdkit.data, atol=1e-1)
    assert X_skfp.shape == (len(mols_conformers_list), 224)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_morse_feature_names():
    morse_fp = MORSEFingerprint()
    feature_names = morse_fp.get_feature_names_out()

    assert len(feature_names) == morse_fp.n_features_out

    assert feature_names[0] == "unweighted 0"
    assert feature_names[1] == "unweighted 1"
    assert feature_names[32] == "atomic mass 0"
    assert feature_names[-1] == "IState 31"
