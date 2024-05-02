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
    usr_fp = USRFingerprint(sparse=False, errors="ignore", n_jobs=-1)
    X_skfp = usr_fp.transform(mols_conformers_three_plus_atoms)

    X_rdkit = []
    for mol in mols_conformers_three_plus_atoms:
        try:
            mol_fp = GetUSR(mol)
            X_rdkit.append(mol_fp)
        except ValueError:
            pass

    X_rdkit = np.array(X_rdkit)

    assert np.abs(X_skfp - X_rdkit).max() < 1e-5


def test_usr_sparse_bit_fingerprint(mols_conformers_three_plus_atoms):
    usr_fp = USRFingerprint(sparse=True, errors="ignore", n_jobs=-1)
    X_skfp = usr_fp.transform(mols_conformers_three_plus_atoms)

    X_rdkit = []
    for mol in mols_conformers_three_plus_atoms:
        try:
            mol_fp = GetUSR(mol)
            X_rdkit.append(mol_fp)
        except ValueError:
            pass

    X_rdkit = csr_array(X_rdkit)

    assert np.abs(X_skfp - X_rdkit).max() < 1e-5


def test_usr_bit_fingerprint_transform_x_y(mols_conformers_three_plus_atoms):
    y = np.arange(len(mols_conformers_three_plus_atoms))

    usr_fp = USRFingerprint(sparse=False, errors="ignore", n_jobs=-1)
    X_skfp, y_skfp = usr_fp.transform_x_y(mols_conformers_three_plus_atoms, y)

    X_rdkit = []
    y_rdkit = []
    for mol, y in zip(mols_conformers_three_plus_atoms, y):
        try:
            mol_fp = GetUSR(mol)
            X_rdkit.append(mol_fp)
            y_rdkit.append(y)
        except ValueError:
            pass

    X_rdkit = np.array(X_rdkit)
    y_rdkit = np.array(y_rdkit)

    assert np.abs(X_skfp - X_rdkit).max() < 1e-5
    assert np.array_equal(y_rdkit, y_skfp)


def test_usr_sparse_bit_fingerprint_transform_x_y(mols_conformers_three_plus_atoms):
    y = np.arange(len(mols_conformers_three_plus_atoms))

    usr_fp = USRFingerprint(sparse=True, errors="ignore", n_jobs=-1)
    X_skfp, y_skfp = usr_fp.transform_x_y(mols_conformers_three_plus_atoms, y)

    X_rdkit = []
    y_rdkit = []
    for mol, y in zip(mols_conformers_three_plus_atoms, y):
        try:
            mol_fp = GetUSR(mol)
            X_rdkit.append(mol_fp)
            y_rdkit.append(y)
        except ValueError:
            pass

    X_rdkit = csr_array(X_rdkit)
    y_rdkit = np.array(y_rdkit)

    assert np.abs(X_skfp - X_rdkit).max() < 1e-5
    assert np.array_equal(y_rdkit, y_skfp)
