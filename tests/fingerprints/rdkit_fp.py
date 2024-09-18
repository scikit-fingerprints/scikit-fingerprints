import numpy as np
import pytest
from rdkit.Chem.rdFingerprintGenerator import (
    GetMorganFeatureAtomInvGen,
    GetRDKitFPGenerator,
)
from scipy.sparse import csr_array
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import RDKitFingerprint


def test_rdkit_bit_fingerprint(smiles_list, mols_list):
    rdkit_fp = RDKitFingerprint(sparse=False, count=False, n_jobs=-1)
    X_skfp = rdkit_fp.transform(smiles_list)

    fp_gen = GetRDKitFPGenerator(countSimulation=False)
    X_rdkit = np.array([fp_gen.GetFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), rdkit_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_rdkit_count_fingerprint(smiles_list, mols_list):
    rdkit_fp = RDKitFingerprint(sparse=False, count=True, n_jobs=-1)
    X_skfp = rdkit_fp.transform(smiles_list)

    fp_gen = GetRDKitFPGenerator(countSimulation=True)
    X_rdkit = np.array([fp_gen.GetCountFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), rdkit_fp.fp_size)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_rdkit_sparse_bit_fingerprint(smiles_list, mols_list):
    rdkit_fp = RDKitFingerprint(sparse=True, count=False, n_jobs=-1)
    X_skfp = rdkit_fp.transform(smiles_list)

    fp_gen = GetRDKitFPGenerator(countSimulation=False)
    X_rdkit = csr_array([fp_gen.GetFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), rdkit_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_rdkit_sparse_count_fingerprint(smiles_list, mols_list):
    rdkit_fp = RDKitFingerprint(sparse=True, count=True, n_jobs=-1)
    X_skfp = rdkit_fp.transform(smiles_list)

    fp_gen = GetRDKitFPGenerator(countSimulation=True)
    X_rdkit = csr_array([fp_gen.GetCountFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), rdkit_fp.fp_size)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)


def test_pharmacophoric_invariants(smiles_list, mols_list):
    pharma_rdkit_fp = RDKitFingerprint(use_pharmacophoric_invariants=True, n_jobs=-1)
    X_skfp = pharma_rdkit_fp.transform(smiles_list)

    fp_gen = GetRDKitFPGenerator(atomInvariantsGenerator=GetMorganFeatureAtomInvGen())
    X_rdkit = np.array([fp_gen.GetFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), pharma_rdkit_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_rdkit_wrong_path_lengths(smiles_list):
    rdkit_fp = RDKitFingerprint(min_path=3, max_path=2)
    with pytest.raises(InvalidParameterError):
        rdkit_fp.transform(smiles_list)
