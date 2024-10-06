from rdkit.Chem import Mol

from skfp.preprocessing import MolFromSmilesTransformer, MolToSmilesTransformer


def test_mol_from_smiles(smiles_list):
    mol_from_smiles = MolFromSmilesTransformer()
    mols_list = mol_from_smiles.transform(smiles_list)

    assert len(mols_list) == len(smiles_list)
    assert all(isinstance(x, Mol) for x in mols_list)


def test_mol_to_smiles(mols_list):
    mol_to_smiles = MolToSmilesTransformer()
    smiles_list = mol_to_smiles.transform(mols_list)

    assert len(smiles_list) == len(mols_list)
    assert all(isinstance(x, str) for x in smiles_list)


def test_mol_to_and_from_smiles(smiles_list):
    mol_from_smiles = MolFromSmilesTransformer()
    mol_to_smiles = MolToSmilesTransformer()

    mols_list = mol_from_smiles.transform(smiles_list)
    smiles_list_2 = mol_to_smiles.transform(mols_list)

    assert smiles_list_2 == smiles_list
