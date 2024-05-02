import numpy as np
import pytest
from rdkit.Chem.rdMolDescriptors import GetUSRCAT
from scipy.sparse import csr_array

from skfp.fingerprints import USRCATFingerprint


@pytest.fixture
def mols_conformers_3_plus_atoms(mols_conformers_list):
    # USRCAT descriptor requires at least 3 atoms to work
    return [mol for mol in mols_conformers_list if mol.GetNumAtoms() >= 3]


def test_usrcat_bit_fingerprint(mols_conformers_3_plus_atoms):
    usrcat_fp = USRCATFingerprint(sparse=False, n_jobs=-1)
    X_skfp = usrcat_fp.transform(mols_conformers_3_plus_atoms)

    X_rdkit = np.array(
        [
            GetUSRCAT(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_3_plus_atoms
        ]
    )

    assert np.allclose(X_skfp, X_rdkit, atol=1e-4)
    assert X_skfp.shape == (len(mols_conformers_3_plus_atoms), 60)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_usrcat_sparse_bit_fingerprint(mols_conformers_3_plus_atoms):
    usrcat_fp = USRCATFingerprint(sparse=True, n_jobs=-1)
    X_skfp = usrcat_fp.transform(mols_conformers_3_plus_atoms)

    X_rdkit = csr_array(
        [
            GetUSRCAT(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_3_plus_atoms
        ]
    )

    assert np.allclose(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(mols_conformers_3_plus_atoms), 60)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_usrcat_bit_fingerprint_transform_x_y(mols_conformers_3_plus_atoms):
    y = np.arange(len(mols_conformers_3_plus_atoms))

    usrcat_fp = USRCATFingerprint(sparse=False, n_jobs=-1)
    X_skfp, y_skfp = usrcat_fp.transform_x_y(mols_conformers_3_plus_atoms, y)

    X_rdkit = []
    y_rdkit = []
    for mol, y in zip(mols_conformers_3_plus_atoms, y):
        mol_fp = GetUSRCAT(mol, confId=mol.GetIntProp("conf_id"))
        X_rdkit.append(mol_fp)
        y_rdkit.append(y)

    X_rdkit = np.array(X_rdkit)
    y_rdkit = np.array(y_rdkit)

    assert np.allclose(X_skfp, X_rdkit, atol=1e-4)
    assert X_skfp.shape == (len(mols_conformers_3_plus_atoms), 60)
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert np.array_equal(y_skfp, y_rdkit)


def test_usrcat_sparse_bit_fingerprint_transform_x_y(mols_conformers_3_plus_atoms):
    y = np.arange(len(mols_conformers_3_plus_atoms))

    usrcat_fp = USRCATFingerprint(sparse=True, n_jobs=-1)
    X_skfp, y_skfp = usrcat_fp.transform_x_y(mols_conformers_3_plus_atoms, y)

    X_rdkit = []
    y_rdkit = []
    for mol, y in zip(mols_conformers_3_plus_atoms, y):
        mol_fp = GetUSRCAT(mol, confId=mol.GetIntProp("conf_id"))
        X_rdkit.append(mol_fp)
        y_rdkit.append(y)

    X_rdkit = csr_array(X_rdkit)
    y_rdkit = np.array(y_rdkit)

    assert np.allclose(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(mols_conformers_3_plus_atoms), 60)
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert np.array_equal(y_rdkit, y_skfp)
