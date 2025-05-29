import numpy as np
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints import PubChemFingerprint


def test_pubchem_bit_fingerprint(smiles_list, mols_list):
    pubchem_fp = PubChemFingerprint(n_jobs=-1)
    X_skfp = pubchem_fp.transform(smiles_list)

    assert X_skfp.shape == (len(mols_list), 881)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_pubchem_count_fingerprint(smiles_list, mols_list):
    pubchem_fp = PubChemFingerprint(count=True, n_jobs=-1)
    X_skfp = pubchem_fp.transform(smiles_list)

    assert X_skfp.shape == (len(mols_list), 757)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_pubchem_sparse_bit_fingerprint(smiles_list, mols_list):
    pubchem_fp = PubChemFingerprint(sparse=True, n_jobs=-1)
    X_skfp = pubchem_fp.transform(smiles_list)

    assert X_skfp.shape == (len(mols_list), 881)
    assert X_skfp.dtype == np.uint8
    assert np.allclose(X_skfp.data, 1)


def test_pubchem_sparse_count_fingerprint(smiles_list, mols_list):
    pubchem_fp = PubChemFingerprint(count=True, sparse=True, n_jobs=-1)
    X_skfp = pubchem_fp.transform(smiles_list)

    assert X_skfp.shape == (len(mols_list), 757)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)


def test_pubchem_patterns():
    pubchem_fp = PubChemFingerprint(count=True)
    mol = MolFromSmiles("[Na+].[Cl-]")
    counts = pubchem_fp._get_pubchem_fingerprint(mol)
    assert len(counts) == 757

    assert counts[0] == 0  # H
    assert counts[7] == 1  # Na
    assert counts[11] == 1  # Cl

    # no rings, no features other than Na and Cl
    assert counts.sum() == 2
