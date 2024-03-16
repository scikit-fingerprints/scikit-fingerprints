import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetRDKitFPGenerator
from scipy.sparse import csr_array

from skfp.fingerprints import RDKitFingerprint


def test_rdkit_bit_fingerprint(smiles_list, mols_list):
    rdkit_fp = RDKitFingerprint(sparse=False, count=False, n_jobs=-1)
    X_skfp = rdkit_fp.transform(smiles_list)

    fp_gen = GetRDKitFPGenerator(countSimulation=False)
    X_rdkit = np.array([fp_gen.GetFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)


def test_rdkit_count_fingerprint(smiles_list, mols_list):
    rdkit_fp = RDKitFingerprint(sparse=False, count=True, n_jobs=-1)
    X_skfp = rdkit_fp.transform(smiles_list)

    fp_gen = GetRDKitFPGenerator(countSimulation=True)
    X_rdkit = np.array([fp_gen.GetCountFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)


def test_rdkit_sparse_bit_fingerprint(smiles_list, mols_list):
    rdkit_fp = RDKitFingerprint(sparse=True, count=False, n_jobs=-1)
    X_skfp = rdkit_fp.transform(smiles_list)

    fp_gen = GetRDKitFPGenerator(countSimulation=False)
    X_rdkit = csr_array([fp_gen.GetFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)


def test_rdkit_sparse_count_fingerprint(smiles_list, mols_list):
    rdkit_fp = RDKitFingerprint(sparse=True, count=True, n_jobs=-1)
    X_skfp = rdkit_fp.transform(smiles_list)

    fp_gen = GetRDKitFPGenerator(countSimulation=True)
    X_rdkit = csr_array([fp_gen.GetCountFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)
