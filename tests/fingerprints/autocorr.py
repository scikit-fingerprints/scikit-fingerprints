import numpy as np
from rdkit.Chem.rdMolDescriptors import CalcAUTOCORR2D, CalcAUTOCORR3D
from scipy.sparse import csr_array

from skfp.fingerprints import AutocorrFingerprint


def test_autocorr_fingerprint(smiles_list, mols_list):
    autocorr_fp = AutocorrFingerprint(use_3D=False, sparse=False, n_jobs=-1)
    X_skfp = autocorr_fp.transform(smiles_list)
    X_rdkit = np.array([CalcAUTOCORR2D(mol) for mol in mols_list])
    assert np.allclose(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), 192)


def test_autocorr_sparse_fingerprint(smiles_list, mols_list):
    autocorr_fp = AutocorrFingerprint(use_3D=False, sparse=True, n_jobs=-1)
    X_skfp = autocorr_fp.transform(smiles_list)
    X_rdkit = csr_array([CalcAUTOCORR2D(mol) for mol in mols_list])
    assert np.allclose(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), 192)


def test_autocorr_3D_fingerprint(mols_conformers_list):
    autocorr_fp = AutocorrFingerprint(use_3D=True, sparse=False, n_jobs=-1)
    X_skfp = autocorr_fp.transform(mols_conformers_list)

    X_rdkit = np.array(
        [CalcAUTOCORR3D(mol, confId=mol.conf_id) for mol in mols_conformers_list]
    )

    assert np.allclose(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(mols_conformers_list), 80)


def test_autocorr_3D_sparse_fingerprint(mols_conformers_list):
    autocorr_fp = AutocorrFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_skfp = autocorr_fp.transform(mols_conformers_list)

    X_rdkit = csr_array(
        [CalcAUTOCORR3D(mol, confId=mol.conf_id) for mol in mols_conformers_list]
    )

    assert np.allclose(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(mols_conformers_list), 80)
