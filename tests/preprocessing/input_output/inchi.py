import numpy as np
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


def test_parallel_to_and_from_inchi(inchi_list):
    mol_from_inchi_seq = MolFromInchiTransformer()
    mol_from_inchi_parallel = MolFromInchiTransformer(n_jobs=-1)

    mol_to_inchi_seq = MolToInchiTransformer()
    mol_to_inchi_parallel = MolToInchiTransformer(n_jobs=-1)

    mols_list_seq = mol_from_inchi_seq.transform(inchi_list)
    mols_list_parallel = mol_from_inchi_parallel.transform(inchi_list)

    inchi_list_2_seq = mol_to_inchi_seq.transform(mols_list_seq)
    inchi_list_2_parallel = mol_to_inchi_parallel.transform(mols_list_parallel)

    assert inchi_list_2_seq == inchi_list
    assert inchi_list_2_seq == inchi_list_2_parallel


def test_from_invalid_inchi(inchi_list):
    invalid_inchi_list = ["noninchi", "invalid"]
    mol_from_inchi = MolFromInchiTransformer(valid_only=False)
    mols_list = mol_from_inchi.transform(inchi_list + invalid_inchi_list)

    mol_from_inchi = MolFromInchiTransformer(valid_only=True)
    mols_list_2 = mol_from_inchi.transform(inchi_list + invalid_inchi_list)

    assert len(mols_list) == len(inchi_list) + len(invalid_inchi_list)
    assert len(mols_list_2) == len(inchi_list)


def test_inchi_transform_x_y(inchi_list):
    labels = np.ones(len(inchi_list))

    mol_from_inchi = MolFromInchiTransformer(valid_only=True)
    mols, labels = mol_from_inchi.transform_x_y(inchi_list, labels)

    assert len(mols) == len(inchi_list)
    assert len(labels) == len(inchi_list)
