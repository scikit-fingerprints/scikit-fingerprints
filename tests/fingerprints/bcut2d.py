import numpy as np
import pytest
from rdkit.Chem.rdMolDescriptors import BCUT2D

from skfp.fingerprints import BCUT2DFingerprint


@pytest.fixture
def gasteiger_allowed_mols(mols_list):
    # Gasteiger partial charge model does not work for metals
    # allowed elements: https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/PartialCharges/GasteigerParams.cpp
    # fmt: off
    allowed_elements = {
        "H", "C", "N", "O", "F", "Cl", "Br", "I", "S", "P", "Si", "B", "Be", "Mg", "Al",
    }
    # fmt: on
    return [
        mol
        for mol in mols_list
        if all(atom.GetSymbol() in allowed_elements for atom in mol.GetAtoms())
    ]


def test_bcut2d_fingerprint_gasteiger(gasteiger_allowed_mols):
    bcut2d_fp = BCUT2DFingerprint(partial_charge_model="Gasteiger", n_jobs=-1)
    X_skfp = bcut2d_fp.transform(gasteiger_allowed_mols)

    X_rdkit = np.array([BCUT2D(mol) for mol in gasteiger_allowed_mols])

    assert np.allclose(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(gasteiger_allowed_mols), 8)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_bcut2d_fingerprint_formal(smallest_mols_list):
    # default formal charge model
    bcut2d_fp = BCUT2DFingerprint(n_jobs=-1)
    X_skfp_parallel = bcut2d_fp.transform(smallest_mols_list)

    bcut2d_fp = BCUT2DFingerprint()
    X_skfp_seq = bcut2d_fp.transform(smallest_mols_list)

    assert np.allclose(X_skfp_parallel, X_skfp_seq)
    assert X_skfp_parallel.shape == X_skfp_seq.shape


def test_bcut2d_ignore_errors(mols_list, gasteiger_allowed_mols):
    bcut2d_fp = BCUT2DFingerprint(
        partial_charge_model="Gasteiger", errors="ignore", n_jobs=-1
    )
    X_skfp = bcut2d_fp.transform(mols_list)
    X_skfp_gasteiger = bcut2d_fp.transform(gasteiger_allowed_mols)

    assert np.allclose(X_skfp, X_skfp_gasteiger)


def test_bcut2d_feature_names():
    bcut2d_fp = BCUT2DFingerprint()
    feature_names = bcut2d_fp.get_feature_names_out()

    assert len(feature_names) == bcut2d_fp.n_features_out
    assert len(feature_names) == len(set(feature_names))

    assert feature_names[0] == "max Burden eigenvalue mass"
    assert feature_names[1] == "min Burden eigenvalue mass"
    assert feature_names[-1] == "min Burden eigenvalue MR"
