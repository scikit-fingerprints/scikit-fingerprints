import numpy as np
import pytest
from rdkit.Chem.rdMolDescriptors import GetUSR
from scipy.sparse import csr_array

from skfp.fingerprints import USRFingerprint


@pytest.fixture
def mols_conformers_three_plus_atoms(mols_conformers_list):
    # USR descriptor requires at least 3 atoms to work
    return [mol for mol in mols_conformers_list if mol.GetNumAtoms() >= 3]


def test_usr_bit_fingerprint(mols_conformers_three_plus_atoms):
    usr_fp = USRFingerprint(sparse=False, n_jobs=-1)
    X_skfp = usr_fp.transform(mols_conformers_three_plus_atoms)

    X_rdkit = np.array(
        [
            GetUSR(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_three_plus_atoms
        ]
    )

    assert np.allclose(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(mols_conformers_three_plus_atoms), 12)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_usr_sparse_bit_fingerprint(mols_conformers_three_plus_atoms):
    usr_fp = USRFingerprint(sparse=True, n_jobs=-1)
    X_skfp = usr_fp.transform(mols_conformers_three_plus_atoms)

    X_rdkit = csr_array(
        [
            GetUSR(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_three_plus_atoms
        ]
    )

    assert np.allclose(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(mols_conformers_three_plus_atoms), 12)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_usr_bit_fingerprint_transform_x_y(mols_conformers_three_plus_atoms):
    y = np.arange(len(mols_conformers_three_plus_atoms))

    usr_fp = USRFingerprint(sparse=False, n_jobs=-1)
    X_skfp, y_skfp = usr_fp.transform_x_y(mols_conformers_three_plus_atoms, y)

    X_rdkit = []
    y_rdkit = []
    for mol, y in zip(mols_conformers_three_plus_atoms, y):
        mol_fp = GetUSR(mol, confId=mol.GetIntProp("conf_id"))
        X_rdkit.append(mol_fp)
        y_rdkit.append(y)

    X_rdkit = np.array(X_rdkit)
    y_rdkit = np.array(y_rdkit)

    assert np.allclose(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(mols_conformers_three_plus_atoms), 12)
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert np.array_equal(y_skfp, y_rdkit)


def test_usr_sparse_bit_fingerprint_transform_x_y(mols_conformers_three_plus_atoms):
    y = np.arange(len(mols_conformers_three_plus_atoms))

    usr_fp = USRFingerprint(sparse=True, n_jobs=-1)
    X_skfp, y_skfp = usr_fp.transform_x_y(mols_conformers_three_plus_atoms, y)

    X_rdkit = []
    y_rdkit = []
    for mol, y in zip(mols_conformers_three_plus_atoms, y):
        mol_fp = GetUSR(mol, confId=mol.GetIntProp("conf_id"))
        X_rdkit.append(mol_fp)
        y_rdkit.append(y)

    X_rdkit = csr_array(X_rdkit)
    y_rdkit = np.array(y_rdkit)

    assert np.allclose(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(mols_conformers_three_plus_atoms), 12)
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert np.array_equal(y_rdkit, y_skfp)
