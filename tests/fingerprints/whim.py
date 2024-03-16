import numpy as np
from fingerprints import WHIMFingerprint
from rdkit.Chem.rdMolDescriptors import CalcWHIM
from scipy.sparse import csr_array


def test_whim_bit_fingerprint(mols_conformers_list):
    whim_fp = WHIMFingerprint(sparse=False, n_jobs=-1)
    X_skfp = whim_fp.transform(mols_conformers_list)

    X_rdkit = np.array(
        [CalcWHIM(mol, confId=mol.conf_id) for mol in mols_conformers_list]
    )

    assert np.all(np.isclose(X_skfp, X_rdkit, atol=1e-1))


def test_whim_sparse_bit_fingerprint(mols_conformers_list):
    whim_fp = WHIMFingerprint(sparse=True, n_jobs=-1)
    X_skfp = whim_fp.transform(mols_conformers_list)

    X_rdkit = csr_array(
        [CalcWHIM(mol, confId=mol.conf_id) for mol in mols_conformers_list]
    )

    assert np.all(np.isclose(X_skfp.data, X_rdkit.data, atol=1e-1))
