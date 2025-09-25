import numpy as np
from numpy.testing import assert_equal
from rdkit.Chem import Mol

from skfp.preprocessing import MolFromSmilesTransformer, MolToSmilesTransformer


def test_mol_from_smiles(smiles_list):
    mol_from_smiles = MolFromSmilesTransformer()
    mols_list = mol_from_smiles.transform(smiles_list)

    assert_equal(len(mols_list), len(smiles_list))
    assert all(isinstance(x, Mol) for x in mols_list)


def test_mol_to_smiles(mols_list):
    mol_to_smiles = MolToSmilesTransformer()
    smiles_list = mol_to_smiles.transform(mols_list)

    assert_equal(len(smiles_list), len(mols_list))
    assert all(isinstance(x, str) for x in smiles_list)


def test_mol_to_and_from_smiles(mols_list):
    mol_from_smiles = MolFromSmilesTransformer()
    mol_to_smiles = MolToSmilesTransformer()

    smiles_list_1 = mol_to_smiles.transform(mols_list)
    smiles_list_2 = mol_to_smiles.transform(
        mol_from_smiles.transform(mol_to_smiles.transform(mols_list))
    )

    assert smiles_list_1 == smiles_list_2


def test_parallel_to_and_from_smiles(mols_list):
    mol_to_smiles_seq = MolToSmilesTransformer()
    mol_to_smiles_parallel = MolToSmilesTransformer(n_jobs=-1)

    smiles_list_seq = mol_to_smiles_seq.transform(mols_list)
    smiles_list_parallel = mol_to_smiles_parallel.transform(mols_list)

    assert smiles_list_seq == smiles_list_parallel


def test_from_invalid_smiles(smiles_list):
    invalid_smiles_list = ["[H]=[H]", "invalid"]
    mol_from_smiles = MolFromSmilesTransformer(valid_only=False)
    mols_list = mol_from_smiles.transform(smiles_list + invalid_smiles_list)

    mol_from_smiles = MolFromSmilesTransformer(valid_only=True)
    mols_list_2 = mol_from_smiles.transform(smiles_list + invalid_smiles_list)

    assert_equal(len(mols_list), len(smiles_list) + len(invalid_smiles_list))
    assert_equal(len(mols_list_2), len(smiles_list))


def test_from_invalid_smiles_with_y(smiles_list):
    invalid_smiles_list = ["[H]=[H]", "invalid"]
    all_smiles_list = smiles_list + invalid_smiles_list
    labels = np.ones(len(all_smiles_list))

    labels[-len(invalid_smiles_list) :] = 0

    mol_from_smiles = MolFromSmilesTransformer(valid_only=False)
    mols_list, y = mol_from_smiles.transform_x_y(all_smiles_list, labels)

    mol_from_smiles = MolFromSmilesTransformer(valid_only=True)
    mols_list_2, y_2 = mol_from_smiles.transform_x_y(all_smiles_list, labels)

    assert_equal(len(mols_list), len(all_smiles_list))
    assert_equal(len(mols_list_2), len(smiles_list))

    assert_equal(len(y), len(all_smiles_list))
    assert_equal(len(y_2), len(smiles_list))
    assert_equal(len(mols_list_2), len(y_2))
