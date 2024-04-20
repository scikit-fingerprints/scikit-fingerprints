import numpy as np
import pytest
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from scipy.sparse import csr_array
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import ERGFingerprint

"""
Note: for some unknown reason passing mols_list from the global fixture sometimes
does not work for ErG fingerprints. This happens only for those tests, and not for
any other fingerprint. Creating molecules from SMILES separately here works.
"""


def test_erg_fuzzy_fingerprint(smiles_list):
    erg_fp = ERGFingerprint(sparse=False, n_jobs=-1)
    X_skfp = erg_fp.transform(smiles_list)

    mols_list = [MolFromSmiles(smi) for smi in smiles_list]
    X_rdkit = np.array([GetErGFingerprint(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), 315)
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert np.all(X_skfp >= 0)


def test_erg_bit_fingerprint(smiles_list):
    erg_fp = ERGFingerprint(variant="bit", sparse=False, n_jobs=-1)
    X_skfp = erg_fp.transform(smiles_list)

    mols_list = [MolFromSmiles(smi) for smi in smiles_list]
    X_rdkit = np.array([GetErGFingerprint(mol, fuzzIncrement=0) for mol in mols_list])
    X_rdkit = (X_rdkit > 0).astype(np.uint8)

    print(X_skfp.shape, X_rdkit.shape)

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), 315)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_erg_count_fingerprint(smiles_list):
    erg_fp = ERGFingerprint(variant="count", sparse=False, n_jobs=-1)
    X_skfp = erg_fp.transform(smiles_list)

    mols_list = [MolFromSmiles(smi) for smi in smiles_list]
    X_rdkit = np.array([GetErGFingerprint(mol, fuzzIncrement=0) for mol in mols_list])
    X_rdkit = X_rdkit.round().astype(np.uint32)

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), 315)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_erg_fuzzy_sparse_fingerprint(smiles_list):
    erg_fp = ERGFingerprint(sparse=True, n_jobs=-1)
    X_skfp = erg_fp.transform(smiles_list)

    mols_list = [MolFromSmiles(smi) for smi in smiles_list]
    X_rdkit = csr_array([GetErGFingerprint(mol) for mol in mols_list])

    assert np.allclose(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), 315)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_erg_bit_sparse_fingerprint(smiles_list):
    erg_fp = ERGFingerprint(variant="bit", sparse=True, n_jobs=-1)
    X_skfp = erg_fp.transform(smiles_list)

    mols_list = [MolFromSmiles(smi) for smi in smiles_list]
    X_rdkit = csr_array([GetErGFingerprint(mol, fuzzIncrement=0) for mol in mols_list])
    X_rdkit.data = X_rdkit.data > 0

    assert np.allclose(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), 315)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_erg_count_sparse_fingerprint(smiles_list):
    erg_fp = ERGFingerprint(variant="count", sparse=True, n_jobs=-1)
    X_skfp = erg_fp.transform(smiles_list)

    mols_list = [MolFromSmiles(smi) for smi in smiles_list]
    X_rdkit = csr_array([GetErGFingerprint(mol, fuzzIncrement=0) for mol in mols_list])
    X_rdkit.data = np.round(X_rdkit.data)

    assert np.allclose(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), 315)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)


def test_erg_wrong_path_lengths(smiles_list):
    erg_fp = ERGFingerprint(min_path=3, max_path=2)
    with pytest.raises(InvalidParameterError):
        erg_fp.transform(smiles_list)
