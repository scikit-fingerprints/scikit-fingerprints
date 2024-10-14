import pytest
from rdkit.Chem import Mol, MolFromSmiles

from skfp.preprocessing.filters.utils import (
    get_max_ring_size,
    get_non_carbon_to_carbon_ratio,
    get_num_carbon_atoms,
    get_num_charged_functional_groups,
    get_num_rigid_bonds,
)


@pytest.fixture
def ibuprofren_mol() -> Mol:
    ibuprofen_smiles: str = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    return MolFromSmiles(ibuprofen_smiles)


def test_get_num_charged_functional_groups(mols_list):
    # just a smoke test - this should not error
    for mol in mols_list:
        get_num_charged_functional_groups(mol)


def test_get_max_sized_ring(ibuprofren_mol):
    assert get_max_ring_size(ibuprofren_mol) == 6


def test_get_number_of_carbons(ibuprofren_mol):
    assert get_num_carbon_atoms(ibuprofren_mol) == 13


def test_get_hc_ratio(ibuprofren_mol):
    assert get_non_carbon_to_carbon_ratio(ibuprofren_mol) == 4.0


def test_get_rigbonds(ibuprofren_mol):
    assert get_num_rigid_bonds(ibuprofren_mol) == 11
