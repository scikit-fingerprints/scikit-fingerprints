import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetTopologicalTorsionGenerator
from scipy.sparse import csr_array

from skfp.fingerprints import TopologicalTorsionFingerprint


def test_topological_torsion_bit_fingerprint(smiles_list, mols_list):
    tt_fp = TopologicalTorsionFingerprint(sparse=False, count=False, n_jobs=-1)
    X_skfp = tt_fp.transform(smiles_list)

    fp_gen = GetTopologicalTorsionGenerator()
    X_rdkit = np.array([fp_gen.GetFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), tt_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_topological_torsion_count_fingerprint(smiles_list, mols_list):
    tt_fp = TopologicalTorsionFingerprint(sparse=False, count=True, n_jobs=-1)
    X_skfp = tt_fp.transform(smiles_list)

    fp_gen = GetTopologicalTorsionGenerator()
    X_rdkit = np.array([fp_gen.GetCountFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), tt_fp.fp_size)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_topological_torsion_sparse_bit_fingerprint(smiles_list, mols_list):
    tt_fp = TopologicalTorsionFingerprint(sparse=True, count=False, n_jobs=-1)
    X_skfp = tt_fp.transform(smiles_list)

    fp_gen = GetTopologicalTorsionGenerator()
    X_rdkit = csr_array([fp_gen.GetFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), tt_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_topological_torsion_sparse_count_fingerprint(smiles_list, mols_list):
    tt_fp = TopologicalTorsionFingerprint(sparse=True, count=True, n_jobs=-1)
    X_skfp = tt_fp.transform(smiles_list)

    fp_gen = GetTopologicalTorsionGenerator()
    X_rdkit = csr_array([fp_gen.GetCountFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), tt_fp.fp_size)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)
