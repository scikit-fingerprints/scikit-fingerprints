import numpy as np
from mordred import Calculator, descriptors
from scipy.sparse import csr_array

from skfp.fingerprints import MordredFingerprint


def test_mordred_fingerprint(smiles_list, mols_list):
    mordred_transformer = MordredFingerprint(sparse=False, n_jobs=-1)
    X_skfp = mordred_transformer.transform(smiles_list)

    calc = Calculator(descriptors, ignore_3D=True)
    X_seq = [calc(mol) for mol in mols_list]
    X_seq = np.array(X_seq, dtype=np.float32)

    assert np.array_equal(X_skfp, X_seq, equal_nan=True)
    assert X_skfp.shape == (len(smiles_list), 1613)


def test_mordred_sparse_fingerprint(smiles_list, mols_list):
    mordred_transformer = MordredFingerprint(sparse=True, n_jobs=-1)
    X_skfp = mordred_transformer.transform(smiles_list)

    calc = Calculator(descriptors, ignore_3D=True)
    X_seq = [calc(mol) for mol in mols_list]
    X_seq = csr_array(X_seq, dtype=np.float32)

    assert np.array_equal(X_skfp.data, X_seq.data, equal_nan=True)
    assert X_skfp.shape == (len(smiles_list), 1613)


def test_mordred_3D_fingerprint(smiles_list, mols_list):
    mordred_transformer = MordredFingerprint(use_3D=True, sparse=False, n_jobs=-1)
    X_skfp = mordred_transformer.transform(smiles_list)

    calc = Calculator(descriptors, ignore_3D=False)
    X_seq = [calc(mol) for mol in mols_list]
    X_seq = np.array(X_seq, dtype=np.float32)

    assert np.array_equal(X_skfp, X_seq, equal_nan=True)
    assert X_skfp.shape == (len(smiles_list), 1826)


def test_mordred_3D_sparse_fingerprint(smiles_list, mols_list):
    mordred_transformer = MordredFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_skfp = mordred_transformer.transform(smiles_list)

    calc = Calculator(descriptors, ignore_3D=False)
    X_seq = [calc(mol) for mol in mols_list]
    X_seq = csr_array(X_seq, dtype=np.float32)

    assert np.array_equal(X_skfp.data, X_seq.data, equal_nan=True)
    assert X_skfp.shape == (len(smiles_list), 1826)
