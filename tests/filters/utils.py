import pytest
from numpy.testing import assert_allclose, assert_equal
from rdkit.Chem import Mol, MolFromSmiles

from skfp.filters.utils import (
    get_max_num_fused_aromatic_rings,
    get_max_ring_size,
    get_non_carbon_to_carbon_ratio,
    get_num_aromatic_rings,
    get_num_carbon_atoms,
    get_num_charged_functional_groups,
    get_num_rigid_bonds,
)


@pytest.fixture
def ibuprofen() -> Mol:
    return MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")


def test_get_num_carbon_atoms(ibuprofen):
    assert_equal(get_num_carbon_atoms(ibuprofen), 13)


def test_get_num_rigid_bonds(ibuprofen):
    assert_equal(get_num_rigid_bonds(ibuprofen), 11)


def test_get_num_aromatic_rings():
    # tetrahydrothiophene
    mol = MolFromSmiles("S1CCCC1")
    assert_equal(get_num_aromatic_rings(mol), 0)

    # benzene
    mol = MolFromSmiles("c1ccccc1")
    assert_equal(get_num_aromatic_rings(mol), 1)

    # naphthalene
    mol = MolFromSmiles("c1c2ccccc2ccc1")
    assert_equal(get_num_aromatic_rings(mol), 2)

    # pyrene
    mol = MolFromSmiles("c1cc2cccc3c2c4c1cccc4cc3")
    assert_equal(get_num_aromatic_rings(mol), 4)

    # chlordiazepoxide
    mol = MolFromSmiles("ClC1=CC2=C(N=C(NC)C[N+]([O-])=C2C3=CC=CC=C3)C=C1")
    assert_equal(get_num_aromatic_rings(mol), 2)


def test_get_max_num_fused_aromatic_rings():
    # tetrahydrothiophene
    mol = MolFromSmiles("S1CCCC1")
    assert_equal(get_max_num_fused_aromatic_rings(mol), 0)

    # benzene
    mol = MolFromSmiles("c1ccccc1")
    assert_equal(get_max_num_fused_aromatic_rings(mol), 0)

    # naphthalene
    mol = MolFromSmiles("c1c2ccccc2ccc1")
    assert_equal(get_max_num_fused_aromatic_rings(mol), 2)

    # pyrene
    mol = MolFromSmiles("c1cc2cccc3c2c4c1cccc4cc3")
    assert_equal(get_max_num_fused_aromatic_rings(mol), 4)

    # chlordiazepoxide
    mol = MolFromSmiles("ClC1=CC2=C(N=C(NC)C[N+]([O-])=C2C3=CC=CC=C3)C=C1")
    assert_equal(get_max_num_fused_aromatic_rings(mol), 0)


def test_get_max_ring_size(ibuprofen):
    assert_equal(get_max_ring_size(ibuprofen), 6)


def test_get_non_carbon_to_carbon_ratio():
    # phenol
    mol = MolFromSmiles("Oc1ccccc1")
    assert_allclose(get_non_carbon_to_carbon_ratio(mol), 1 / 6)


def test_get_num_charged_functional_groups(mols_list):
    # just a smoke test - this should not error
    for mol in mols_list:
        get_num_charged_functional_groups(mol)
