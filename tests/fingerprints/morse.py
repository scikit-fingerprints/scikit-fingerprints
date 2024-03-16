import numpy as np
from fingerprints import MORSEFingerprint
from rdkit.Chem.rdMolDescriptors import CalcMORSE
from scipy.sparse import csr_array


def test_morse_bit_fingerprint(mols_conformers_list):
    morse_fp = MORSEFingerprint(sparse=False, n_jobs=-1)
    X_skfp = morse_fp.transform(mols_conformers_list)

    X_rdkit = np.array([CalcMORSE(mol) for mol in mols_conformers_list])

    assert np.all(np.isclose(X_skfp, X_rdkit, rtol=1e-1))


def test_morse_sparse_bit_fingerprint(mols_conformers_list):
    morse_fp = MORSEFingerprint(sparse=True, n_jobs=-1)
    X_skfp = morse_fp.transform(mols_conformers_list)

    X_rdkit = csr_array(
        [CalcMORSE(mol, confId=mol.conf_id) for mol in mols_conformers_list]
    )

    assert np.all(np.isclose(X_skfp.data, X_rdkit.data, atol=1e-1))
