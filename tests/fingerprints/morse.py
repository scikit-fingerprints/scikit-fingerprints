import numpy as np
from rdkit.Chem.rdMolDescriptors import CalcMORSE
from scipy.sparse import csr_array

from skfp.fingerprints import MORSEFingerprint


def test_morse_fingerprint(mols_conformers_list):
    morse_fp = MORSEFingerprint(sparse=False, n_jobs=-1)
    X_skfp = morse_fp.transform(mols_conformers_list)

    X_rdkit = np.array(
        [CalcMORSE(mol, confId=mol.conf_id) for mol in mols_conformers_list]
    )

    assert np.allclose(X_skfp, X_rdkit, atol=1e-1)
    assert X_skfp.shape == (len(mols_conformers_list), 224)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_morse_sparse_fingerprint(mols_conformers_list):
    morse_fp = MORSEFingerprint(sparse=True, n_jobs=-1)
    X_skfp = morse_fp.transform(mols_conformers_list)

    X_rdkit = csr_array(
        [CalcMORSE(mol, confId=mol.conf_id) for mol in mols_conformers_list]
    )

    assert np.allclose(X_skfp.data, X_rdkit.data, atol=1e-1)
    assert X_skfp.shape == (len(mols_conformers_list), 224)
    assert np.issubdtype(X_skfp.dtype, np.floating)
