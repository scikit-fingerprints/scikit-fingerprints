import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from rdkit.Chem.rdFingerprintGenerator import (
    GetAtomPairGenerator,
    GetMorganFeatureAtomInvGen,
)
from scipy.sparse import csr_array
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import AtomPairFingerprint


def test_atom_pair_bit_fingerprint(smiles_list, mols_list):
    atom_pair_fp = AtomPairFingerprint(n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list)

    fp_gen = GetAtomPairGenerator()
    X_rdkit = np.array([fp_gen.GetFingerprintAsNumPy(mol) for mol in mols_list])

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(smiles_list), atom_pair_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_atom_pair_bit_3D_fingerprint(mols_conformers_list):
    atom_pair_fp = AtomPairFingerprint(use_3D=True, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(mols_conformers_list)

    fp_gen = GetAtomPairGenerator(use2D=False)
    X_rdkit = np.array(
        [
            fp_gen.GetFingerprintAsNumPy(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_list
        ]
    )

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), atom_pair_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_atom_pair_count_fingerprint(smiles_list, mols_list):
    atom_pair_fp = AtomPairFingerprint(count=True, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list)

    fp_gen = GetAtomPairGenerator()
    X_rdkit = np.array([fp_gen.GetCountFingerprintAsNumPy(mol) for mol in mols_list])

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(smiles_list), atom_pair_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_atom_pair_count_3D_fingerprint(mols_conformers_list):
    atom_pair_fp = AtomPairFingerprint(use_3D=True, count=True, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(mols_conformers_list)

    fp_gen = GetAtomPairGenerator(use2D=False)
    X_rdkit = np.array(
        [
            fp_gen.GetCountFingerprintAsNumPy(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_list
        ]
    )

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), atom_pair_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_atom_pair_sparse_bit_fingerprint(smiles_list, mols_list):
    atom_pair_fp = AtomPairFingerprint(sparse=True, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list)

    fp_gen = GetAtomPairGenerator()
    X_rdkit = csr_array([fp_gen.GetFingerprintAsNumPy(mol) for mol in mols_list])

    assert_equal(X_skfp.data, X_rdkit.data)
    assert_equal(X_skfp.shape, (len(smiles_list), atom_pair_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_atom_pair_sparse_3D_bit_fingerprint(mols_conformers_list):
    atom_pair_fp = AtomPairFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(mols_conformers_list)

    fp_gen = GetAtomPairGenerator(use2D=False)
    X_rdkit = csr_array(
        [
            fp_gen.GetFingerprintAsNumPy(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_list
        ]
    )

    assert_equal(X_skfp.data, X_rdkit.data)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), atom_pair_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_atom_pair_sparse_count_fingerprint(smiles_list, mols_list):
    atom_pair_fp = AtomPairFingerprint(sparse=True, count=True, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list)

    fp_gen = GetAtomPairGenerator()
    X_rdkit = csr_array([fp_gen.GetCountFingerprintAsNumPy(mol) for mol in mols_list])

    assert_equal(X_skfp.data, X_rdkit.data)
    assert_equal(X_skfp.shape, (len(smiles_list), atom_pair_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)


def test_atom_pair_sparse_3D_count_fingerprint(mols_conformers_list):
    atom_pair_fp = AtomPairFingerprint(use_3D=True, sparse=True, count=True, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(mols_conformers_list)

    fp_gen = GetAtomPairGenerator(use2D=False)
    X_rdkit = csr_array(
        [
            fp_gen.GetCountFingerprintAsNumPy(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_list
        ]
    )

    assert_equal(X_skfp.data, X_rdkit.data)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), atom_pair_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)


def test_pharmacophoric_invariants(smiles_list, mols_list):
    pharma_ap_fp = AtomPairFingerprint(use_pharmacophoric_invariants=True, n_jobs=-1)
    X_skfp = pharma_ap_fp.transform(smiles_list)

    fp_gen = GetAtomPairGenerator(atomInvariantsGenerator=GetMorganFeatureAtomInvGen())
    X_rdkit = np.array([fp_gen.GetFingerprintAsNumPy(mol) for mol in mols_list])

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(smiles_list), pharma_ap_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_atom_pair_hac_scaling(smiles_list, mols_list):
    atom_pair_fp = AtomPairFingerprint(scale_by_hac=True, count=True, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list)

    fp_gen = GetAtomPairGenerator()
    X_rdkit = np.array([fp_gen.GetCountFingerprintAsNumPy(mol) for mol in mols_list])
    X_rdkit_scaled = [
        (100 * fp) / mol.GetNumHeavyAtoms()
        for fp, mol in zip(X_rdkit, mols_list, strict=False)
    ]

    assert_allclose(X_skfp, X_rdkit_scaled)
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert_equal(X_skfp.shape, (len(smiles_list), atom_pair_fp.fp_size))


def test_atom_pair_squared_hac_scaling(smiles_list, mols_list):
    atom_pair_fp = AtomPairFingerprint(scale_by_hac=2, count=True, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list)

    fp_gen = GetAtomPairGenerator()
    X_rdkit = np.array([fp_gen.GetCountFingerprintAsNumPy(mol) for mol in mols_list])
    X_rdkit_scaled = [
        (100 * fp) / (mol.GetNumHeavyAtoms() ** 2)
        for fp, mol in zip(X_rdkit, mols_list, strict=False)
    ]

    assert_allclose(X_skfp, X_rdkit_scaled)
    assert np.all((X_skfp >= 0) & (X_skfp <= 100))
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert_equal(X_skfp.shape, (len(smiles_list), atom_pair_fp.fp_size))


def test_hac_scaling_empty_mol():
    # empty molecule will get division by zero, which is expected, since HAC is zero
    smiles_list = [""]
    atom_pair_fp = AtomPairFingerprint(count=True, scale_by_hac=True)
    with pytest.raises(ZeroDivisionError):
        atom_pair_fp.transform(smiles_list)


def test_atom_pair_wrong_distances(smiles_list):
    atom_pair_fp = AtomPairFingerprint(min_distance=3, max_distance=2)
    with pytest.raises(InvalidParameterError):
        atom_pair_fp.transform(smiles_list)
