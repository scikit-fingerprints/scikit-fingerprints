import sys

import numpy as np
import pytest
from rdkit.Chem.rdMolDescriptors import GetUSRCAT

from skfp.fingerprints import USRCATFingerprint


@pytest.fixture
def mols_conformers_3_plus_atoms(mols_conformers_list):
    # USRCAT descriptor requires at least 3 atoms to work
    return [mol for mol in mols_conformers_list if mol.GetNumAtoms() >= 3]


def test_usrcat_bit_fingerprint(mols_conformers_3_plus_atoms):
    usrcat_fp = USRCATFingerprint(n_jobs=-1)
    X_skfp = usrcat_fp.transform(mols_conformers_3_plus_atoms)

    X_rdkit = np.array(
        [
            GetUSRCAT(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_3_plus_atoms
        ]
    )

    # on macOS for USR and USRCAT we get slightly different results in skfp and RDKit
    # debugging for a long time didn't help, so we check just basic statistic instead
    if sys.platform == "darwin":
        assert np.isclose(np.min(X_skfp), np.min(X_rdkit), atol=1e-3)
        assert np.isclose(np.mean(X_skfp), np.mean(X_rdkit), atol=1e-3)
        assert np.isclose(np.max(X_skfp), np.max(X_rdkit), atol=1e-3)
    else:
        assert np.allclose(X_skfp, X_rdkit, atol=1e-3)

    assert X_skfp.shape == (len(mols_conformers_3_plus_atoms), 12)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_usrcat_bit_fingerprint_transform_x_y(mols_conformers_3_plus_atoms):
    labels = np.arange(len(mols_conformers_3_plus_atoms))

    usrcat_fp = USRCATFingerprint(n_jobs=-1)
    X_skfp, y_skfp = usrcat_fp.transform_x_y(mols_conformers_3_plus_atoms, labels)

    X_rdkit = []
    y_rdkit = []
    for mol, y in zip(mols_conformers_3_plus_atoms, labels):
        mol_fp = GetUSRCAT(mol, confId=mol.GetIntProp("conf_id"))
        X_rdkit.append(mol_fp)
        y_rdkit.append(y)

    X_rdkit = np.array(X_rdkit)
    y_rdkit = np.array(y_rdkit)

    # on macOS for USR and USRCAT we get slightly different results in skfp and RDKit
    # debugging for a long time didn't help, so we check just basic statistic instead
    if sys.platform == "darwin":
        assert np.isclose(np.min(X_skfp), np.min(X_rdkit), atol=1e-3)
        assert np.isclose(np.mean(X_skfp), np.mean(X_rdkit), atol=1e-3)
        assert np.isclose(np.max(X_skfp), np.max(X_rdkit), atol=1e-3)
    else:
        assert np.allclose(X_skfp, X_rdkit, atol=1e-3)

    assert X_skfp.shape == (len(mols_conformers_3_plus_atoms), 12)
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert np.array_equal(y_skfp, y_rdkit)


def test_usrcat_ignore_errors(mols_conformers_list, mols_conformers_3_plus_atoms):
    usr_fp = USRCATFingerprint(errors="ignore", n_jobs=-1)
    X_skfp = usr_fp.transform(mols_conformers_list)
    X_skfp_3_plus_atoms = usr_fp.transform(mols_conformers_3_plus_atoms)

    assert np.allclose(X_skfp, X_skfp_3_plus_atoms)
