import numpy as np
import pytest
from rdkit.Chem.rdMolDescriptors import GetUSRCAT

from skfp.fingerprints import USRCATFingerprint


@pytest.fixture
def mols_conformers_3_plus_atoms(mols_conformers_list):
    # USRCAT descriptor requires at least 3 atoms to work
    return [mol for mol in mols_conformers_list if mol.GetNumAtoms() >= 3]


def test_usrcat_bit_fingerprint(mols_conformers_3_plus_atoms):
    usrcat_fp = USRCATFingerprint()
    X_skfp = usrcat_fp.transform(mols_conformers_3_plus_atoms)

    X_rdkit = np.array(
        [
            GetUSRCAT(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_3_plus_atoms
        ]
    )

    assert np.allclose(X_skfp, X_rdkit, atol=1e-3)
    assert X_skfp.shape == (len(mols_conformers_3_plus_atoms), 60)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_usrcat_bit_fingerprint_transform_x_y(mols_conformers_3_plus_atoms):
    labels = np.arange(len(mols_conformers_3_plus_atoms))

    usrcat_fp = USRCATFingerprint()
    X_skfp, y_skfp = usrcat_fp.transform_x_y(mols_conformers_3_plus_atoms, labels)

    X_rdkit = []
    y_rdkit = []
    for mol, y in zip(mols_conformers_3_plus_atoms, labels, strict=False):
        mol_fp = GetUSRCAT(mol, confId=mol.GetIntProp("conf_id"))
        X_rdkit.append(mol_fp)
        y_rdkit.append(y)

    X_rdkit = np.array(X_rdkit)
    y_rdkit = np.array(y_rdkit)

    assert np.allclose(X_skfp, X_rdkit, atol=1e-3)
    assert X_skfp.shape == (len(mols_conformers_3_plus_atoms), 60)
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert np.array_equal(y_skfp, y_rdkit)


def test_usrcat_ignore_errors(mols_conformers_list, mols_conformers_3_plus_atoms):
    usrcat_fp = USRCATFingerprint(errors="ignore", n_jobs=-1)
    X_skfp = usrcat_fp.transform(mols_conformers_list)
    X_skfp_3_plus_atoms = usrcat_fp.transform(mols_conformers_3_plus_atoms)

    assert np.allclose(X_skfp, X_skfp_3_plus_atoms)


def test_usrcat_copy(mols_conformers_3_plus_atoms):
    # smoke test, should not throw an error
    labels = np.ones(len(mols_conformers_3_plus_atoms))

    usrcat_fp = USRCATFingerprint(errors="ignore", n_jobs=-1)
    fps, labels_out = usrcat_fp.transform_x_y(
        mols_conformers_3_plus_atoms, labels, copy=True
    )

    assert np.array_equal(labels, labels_out)
    assert labels is not labels_out
