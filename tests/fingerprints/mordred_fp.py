import numpy as np
from mordred import Calculator, descriptors
from scipy.sparse import csr_array

from skfp.fingerprints import MordredFingerprint


def test_mordred_fingerprint(smallest_smiles_list, smallest_mols_list):
    mordred_transformer = MordredFingerprint(n_jobs=-1)
    X_skfp = mordred_transformer.transform(smallest_smiles_list)

    calc = Calculator(descriptors, ignore_3D=True)
    X_seq = [calc(mol) for mol in smallest_mols_list]
    X_seq = np.array(X_seq, dtype=np.float32)

    assert np.array_equal(X_skfp, X_seq, equal_nan=True)
    assert X_skfp.shape == (len(smallest_smiles_list), 1613)
    assert X_skfp.dtype == np.float32


def test_mordred_sparse_fingerprint(smallest_smiles_list, smallest_mols_list):
    mordred_transformer = MordredFingerprint(sparse=True, n_jobs=-1)
    X_skfp = mordred_transformer.transform(smallest_smiles_list)

    calc = Calculator(descriptors, ignore_3D=True)
    X_seq = [calc(mol) for mol in smallest_mols_list]
    X_seq = csr_array(X_seq, dtype=np.float32)

    assert np.array_equal(X_skfp.data, X_seq.data, equal_nan=True)
    assert X_skfp.shape == (len(smallest_smiles_list), 1613)
    assert X_skfp.dtype == np.float32


def test_mordred_3D_fingerprint(smallest_smiles_list, smallest_mols_list):
    mordred_transformer = MordredFingerprint(use_3D=True, n_jobs=-1)
    X_skfp = mordred_transformer.transform(smallest_smiles_list)

    calc = Calculator(descriptors, ignore_3D=False)
    X_seq = [calc(mol) for mol in smallest_mols_list]
    X_seq = np.array(X_seq, dtype=np.float32)

    assert np.array_equal(X_skfp, X_seq, equal_nan=True)
    assert X_skfp.shape == (len(smallest_smiles_list), 1826)
    assert X_skfp.dtype == np.float32


def test_mordred_3D_sparse_fingerprint(smallest_smiles_list, smallest_mols_list):
    mordred_transformer = MordredFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_skfp = mordred_transformer.transform(smallest_smiles_list)

    calc = Calculator(descriptors, ignore_3D=False)
    X_seq = [calc(mol) for mol in smallest_mols_list]
    X_seq = csr_array(X_seq, dtype=np.float32)

    assert np.array_equal(X_skfp.data, X_seq.data, equal_nan=True)
    assert X_skfp.shape == (len(smallest_smiles_list), 1826)
    assert X_skfp.dtype == np.float32


def test_mordred_feature_names():
    mordred_transformer = MordredFingerprint()
    calc = Calculator(descriptors, ignore_3D=True)
    feature_names = mordred_transformer.get_feature_names_out()
    assert np.array_equal(feature_names, list(str(d) for d in calc.descriptors))


def test_mordred_3D_feature_names():
    mordred_transformer = MordredFingerprint(use_3D=True)
    calc = Calculator(descriptors, ignore_3D=False)
    feature_names = mordred_transformer.get_feature_names_out()
    assert np.array_equal(feature_names, list(str(d) for d in calc.descriptors))
