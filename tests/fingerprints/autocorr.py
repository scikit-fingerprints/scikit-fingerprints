import numpy as np
from fingerprints import AutocorrFingerprint
from rdkit.Chem.rdMolDescriptors import CalcAUTOCORR2D
from scipy.sparse import csr_array


def test_autocorr_bit_fingerprint(smiles_list, mols_list):
    autocorr_fp = AutocorrFingerprint(sparse=False, n_jobs=-1)
    X_skfp = autocorr_fp.transform(smiles_list)
    X_rdkit = np.array([CalcAUTOCORR2D(mol) for mol in mols_list])
    assert np.array_equal(X_skfp, X_rdkit)


def test_autocorr_sparse_bit_fingerprint(smiles_list, mols_list):
    autocorr_fp = AutocorrFingerprint(sparse=True, n_jobs=-1)
    X_skfp = autocorr_fp.transform(smiles_list)
    X_rdkit = csr_array([CalcAUTOCORR2D(mol) for mol in mols_list])
    assert np.array_equal(X_skfp.data, X_rdkit.data)
