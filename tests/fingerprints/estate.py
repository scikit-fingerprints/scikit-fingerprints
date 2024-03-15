import numpy as np
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from scipy.sparse import csr_array

from skfp.fingerprints import EStateFingerprint


def test_estate_bit_fingerprint(smiles_list, mols_list):
    estate_fp = EStateFingerprint(variant="bit", sparse=False, n_jobs=-1)
    X_skfp = estate_fp.transform(smiles_list)

    X_rdkit = np.array([FingerprintMol(mol) for mol in mols_list])
    X_rdkit = X_rdkit[:, 0] > 0

    assert np.array_equal(X_skfp, X_rdkit)


def test_estate_count_fingerprint(smiles_list, mols_list):
    estate_fp = EStateFingerprint(variant="count", sparse=False, n_jobs=-1)
    X_skfp = estate_fp.transform(smiles_list)

    X_rdkit = np.array([FingerprintMol(mol) for mol in mols_list])
    X_rdkit = X_rdkit[:, 0]

    assert np.array_equal(X_skfp, X_rdkit)


def test_estate_sum_fingerprint(smiles_list, mols_list):
    estate_fp = EStateFingerprint(variant="sum", sparse=False, n_jobs=-1)
    X_skfp = estate_fp.transform(smiles_list)

    X_rdkit = np.array([FingerprintMol(mol) for mol in mols_list])
    X_rdkit = X_rdkit[:, 1]

    assert np.array_equal(X_skfp, X_rdkit)


def test_estate_sparse_bit_fingerprint(smiles_list, mols_list):
    estate_fp = EStateFingerprint(variant="bit", sparse=True, n_jobs=-1)
    X_skfp = estate_fp.transform(smiles_list)

    X_rdkit = np.array([FingerprintMol(mol) for mol in mols_list])
    X_rdkit = X_rdkit[:, 0] > 0
    X_rdkit = csr_array(X_rdkit)

    assert np.array_equal(X_skfp.data, X_rdkit.data)


def test_estate_sparse_count_fingerprint(smiles_list, mols_list):
    estate_fp = EStateFingerprint(variant="count", sparse=True, n_jobs=-1)
    X_skfp = estate_fp.transform(smiles_list)

    X_rdkit = np.array([FingerprintMol(mol) for mol in mols_list])
    X_rdkit = X_rdkit[:, 0]
    X_rdkit = csr_array(X_rdkit)

    assert np.array_equal(X_skfp.data, X_rdkit.data)


def test_estate_sparse_sum_fingerprint(smiles_list, mols_list):
    estate_fp = EStateFingerprint(variant="sum", sparse=True, n_jobs=-1)
    X_skfp = estate_fp.transform(smiles_list)

    X_rdkit = np.array([FingerprintMol(mol) for mol in mols_list])
    X_rdkit = X_rdkit[:, 1]
    X_rdkit = csr_array(X_rdkit)

    assert np.array_equal(X_skfp.data, X_rdkit.data)
