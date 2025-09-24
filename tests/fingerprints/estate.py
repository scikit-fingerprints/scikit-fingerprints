import numpy as np
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from scipy.sparse import csr_array

from skfp.fingerprints import EStateFingerprint


def test_estate_bit_fingerprint(smiles_list, mols_list):
    estate_fp = EStateFingerprint(variant="bit", n_jobs=-1)
    X_skfp = estate_fp.transform(smiles_list)

    X_rdkit = np.array([FingerprintMol(mol) for mol in mols_list])
    X_rdkit = X_rdkit[:, 0] > 0

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), 79)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_estate_count_fingerprint(smiles_list, mols_list):
    estate_fp = EStateFingerprint(variant="count", n_jobs=-1)
    X_skfp = estate_fp.transform(smiles_list)

    X_rdkit = np.array([FingerprintMol(mol) for mol in mols_list])
    X_rdkit = X_rdkit[:, 0]

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), 79)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_estate_sum_fingerprint(smiles_list, mols_list):
    estate_fp = EStateFingerprint(variant="sum", n_jobs=-1)
    X_skfp = estate_fp.transform(smiles_list)

    X_rdkit = np.array([FingerprintMol(mol) for mol in mols_list])
    X_rdkit = X_rdkit[:, 1]

    np.testing.assert_allclose(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), 79)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_estate_sparse_bit_fingerprint(smiles_list, mols_list):
    estate_fp = EStateFingerprint(variant="bit", sparse=True, n_jobs=-1)
    X_skfp = estate_fp.transform(smiles_list)

    X_rdkit = np.array([FingerprintMol(mol) for mol in mols_list])
    X_rdkit = X_rdkit[:, 0] > 0
    X_rdkit = csr_array(X_rdkit)

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), 79)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_estate_sparse_count_fingerprint(smiles_list, mols_list):
    estate_fp = EStateFingerprint(variant="count", sparse=True, n_jobs=-1)
    X_skfp = estate_fp.transform(smiles_list)

    X_rdkit = np.array([FingerprintMol(mol) for mol in mols_list])
    X_rdkit = X_rdkit[:, 0]
    X_rdkit = csr_array(X_rdkit)

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), 79)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)


def test_estate_sparse_sum_fingerprint(smiles_list, mols_list):
    estate_fp = EStateFingerprint(variant="sum", sparse=True, n_jobs=-1)
    X_skfp = estate_fp.transform(smiles_list)

    X_rdkit = np.array([FingerprintMol(mol) for mol in mols_list])
    X_rdkit = X_rdkit[:, 1]
    X_rdkit = csr_array(X_rdkit)

    np.testing.assert_allclose(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), 79)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_estate_feature_names():
    estate_fp = EStateFingerprint()
    feature_names = estate_fp.get_feature_names_out()

    assert len(feature_names) == estate_fp.n_features_out
    assert len(feature_names) == len(set(feature_names))

    assert feature_names[0] == "[LiD1]-*"
    assert feature_names[-1] == "[PbD4H0](-*)(-*)(-*)-*"
