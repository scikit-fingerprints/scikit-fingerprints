import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.rdMolDescriptors import GetUSR, GetUSRCAT
from scipy.sparse import csr_array

from skfp.fingerprints import USRDescriptor


def test_usr_bit_fingerprint(mols_conformers_list):
    usr_fp = USRDescriptor(sparse=False, errors="ignore", n_jobs=-1)
    X_skfp = usr_fp.transform(mols_conformers_list)

    X_rdkit = []
    for mol in mols_conformers_list:
        try:
            mol_fp = GetUSR(mol)
            X_rdkit.append(mol_fp)
        except ValueError:
            pass

    X_rdkit = np.array(X_rdkit)

    assert np.abs(X_skfp - X_rdkit).max() < 1e-5


def test_usr_sparse_bit_fingerprint(mols_conformers_list):
    usr_fp = USRDescriptor(sparse=True, errors="ignore", n_jobs=-1)
    X_skfp = usr_fp.transform(mols_conformers_list)

    X_rdkit = []
    for mol in mols_conformers_list:
        try:
            mol_fp = GetUSR(mol)
            X_rdkit.append(mol_fp)
        except ValueError:
            pass

    X_rdkit = csr_array(X_rdkit)

    assert np.abs(X_skfp - X_rdkit).max() < 1e-5


def test_usr_bit_fingerprint_transform_x_y(mols_conformers_list):
    fake_labels = np.arange(len(mols_conformers_list))

    usr_fp = USRDescriptor(sparse=False, errors="ignore", n_jobs=-1)
    X_skfp, y_skfp = usr_fp.transform_x_y(mols_conformers_list, fake_labels)

    X_rdkit = []
    y_rdkit = []
    for mol, y in zip(mols_conformers_list, fake_labels):
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


def test_usr_sparse_bit_fingerprint_transform_x_y(mols_conformers_list):
    fake_labels = np.arange(len(mols_conformers_list))

    usr_fp = USRDescriptor(sparse=True, errors="ignore", n_jobs=-1)
    X_skfp, y_skfp = usr_fp.transform_x_y(mols_conformers_list, fake_labels)

    X_rdkit = []
    y_rdkit = []
    for mol, y in zip(mols_conformers_list, fake_labels):
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


def test_usr_cat_bit_fingerprint(mols_conformers_list):
    usr_fp = USRDescriptor(sparse=False, errors="ignore", use_usr_cat=True, n_jobs=-1)
    X_skfp = usr_fp.transform(mols_conformers_list)

    X_rdkit = []
    for mol in mols_conformers_list:
        try:
            mol_fp = GetUSRCAT(mol)
            X_rdkit.append(mol_fp)
        except ValueError:
            pass

    X_rdkit = np.array(X_rdkit)

    assert np.abs(X_skfp - X_rdkit).max() < 1e-4


def test_usr_cat_sparse_bit_fingerprint(mols_conformers_list):
    usr_fp = USRDescriptor(sparse=True, errors="ignore", use_usr_cat=True, n_jobs=-1)
    X_skfp = usr_fp.transform(mols_conformers_list)

    X_rdkit = []
    for mol in mols_conformers_list:
        try:
            mol_fp = GetUSRCAT(mol)
            X_rdkit.append(mol_fp)
        except ValueError:
            pass

    X_rdkit = csr_array(X_rdkit)

    assert np.abs(X_skfp - X_rdkit).max() < 1e-4


def test_usr_cat_bit_fingerprint_transform_x_y(mols_conformers_list):
    fake_labels = np.arange(len(mols_conformers_list))

    usr_fp = USRDescriptor(sparse=False, errors="ignore", use_usr_cat=True, n_jobs=-1)
    X_skfp, y_skfp = usr_fp.transform_x_y(mols_conformers_list, fake_labels)

    X_rdkit = []
    y_rdkit = []
    for mol, y in zip(mols_conformers_list, fake_labels):
        try:
            mol_fp = GetUSRCAT(mol)
            X_rdkit.append(mol_fp)
            y_rdkit.append(y)
        except ValueError:
            pass

    X_rdkit = np.array(X_rdkit)
    y_rdkit = np.array(y_rdkit)

    assert np.abs(X_skfp - X_rdkit).max() < 1e-4
    assert np.array_equal(y_rdkit, y_skfp)


def test_usr_cat_sparse_bit_fingerprint_transform_x_y(mols_conformers_list):
    fake_labels = np.arange(len(mols_conformers_list))

    usr_fp = USRDescriptor(sparse=True, errors="ignore", use_usr_cat=True, n_jobs=-1)
    X_skfp, y_skfp = usr_fp.transform_x_y(mols_conformers_list, fake_labels)

    X_rdkit = []
    y_rdkit = []
    for mol, y in zip(mols_conformers_list, fake_labels):
        try:
            mol_fp = GetUSRCAT(mol)
            X_rdkit.append(mol_fp)
            y_rdkit.append(y)
        except ValueError:
            pass

    X_rdkit = csr_array(X_rdkit)
    y_rdkit = np.array(y_rdkit)

    assert np.abs(X_skfp - X_rdkit).max() < 1e-4
    assert np.array_equal(y_rdkit, y_skfp)
