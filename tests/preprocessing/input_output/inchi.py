import pytest
from rdkit.Chem import Mol, MolFromSmiles, MolToInchi

from skfp.preprocessing import MolFromInchiTransformer, MolToInchiTransformer


@pytest.fixture
def inchi_list(smiles_list):
    return [MolToInchi(MolFromSmiles(smi)) for smi in smiles_list]


def test_mol_from_inchi(inchi_list):
    mol_from_inchi = MolFromInchiTransformer()
    mols_list = mol_from_inchi.transform(inchi_list)

    assert len(mols_list) == len(inchi_list)
    assert all(isinstance(x, Mol) for x in mols_list)


def test_mol_to_inchi(mols_list):
    mol_to_inchi = MolToInchiTransformer()
    inchi_list = mol_to_inchi.transform(mols_list)

    assert len(inchi_list) == len(mols_list)
    assert all(isinstance(x, str) for x in inchi_list)


def test_mol_to_and_from_inchi(inchi_list):
    mol_from_inchi = MolFromInchiTransformer()
    mol_to_inchi = MolToInchiTransformer()

    mols_list = mol_from_inchi.transform(inchi_list)
    inchi_list_2 = mol_to_inchi.transform(mols_list)

    assert inchi_list_2 == inchi_list
