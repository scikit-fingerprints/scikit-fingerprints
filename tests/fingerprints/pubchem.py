import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints import PubChemFingerprint


def test_pubchem_bit_fingerprint(smiles_list, mols_list):
    pubchem_fp = PubChemFingerprint(n_jobs=-1)
    X_skfp = pubchem_fp.transform(smiles_list)

    assert_equal(X_skfp.shape, (len(mols_list), 881))
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_pubchem_count_fingerprint(smiles_list, mols_list):
    pubchem_fp = PubChemFingerprint(count=True, n_jobs=-1)
    X_skfp = pubchem_fp.transform(smiles_list)

    assert_equal(X_skfp.shape, (len(mols_list), 757))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_pubchem_sparse_bit_fingerprint(smiles_list, mols_list):
    pubchem_fp = PubChemFingerprint(sparse=True, n_jobs=-1)
    X_skfp = pubchem_fp.transform(smiles_list)

    assert_equal(X_skfp.shape, (len(mols_list), 881))
    assert X_skfp.dtype == np.uint8
    assert_allclose(X_skfp.data, 1)


def test_pubchem_sparse_count_fingerprint(smiles_list, mols_list):
    pubchem_fp = PubChemFingerprint(count=True, sparse=True, n_jobs=-1)
    X_skfp = pubchem_fp.transform(smiles_list)

    assert_equal(X_skfp.shape, (len(mols_list), 757))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)


def test_pubchem_patterns():
    pubchem_fp = PubChemFingerprint(count=True)
    mol = MolFromSmiles("[Na+].[Cl-]")
    counts = pubchem_fp._get_pubchem_fingerprint(mol)
    assert_equal(len(counts), 757)

    assert_equal(counts[0], 0)  # H
    assert_equal(counts[7], 1)  # Na
    assert_equal(counts[11], 1)  # Cl

    # no rings, no features other than Na and Cl
    assert_equal(counts.sum(), 2)


@pytest.mark.parametrize("fp", [PubChemFingerprint(), PubChemFingerprint(count=True)])
def test_pubchem_feature_names(fp):
    feature_names = fp.get_feature_names_out()
    assert_equal(len(feature_names), fp.n_features_out)
    assert_equal(len(feature_names), len(set(feature_names)))

    assert feature_names[0].startswith("H")
    assert_equal(feature_names[-1], "Br[#6]1[#6](Br)[#6][#6][#6]1")
