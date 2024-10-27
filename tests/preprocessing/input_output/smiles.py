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


def test_parallel_to_and_from_smiles(smiles_list):
    mol_from_smiles_seq = MolFromSmilesTransformer()
    mol_from_smiles_parallel = MolFromSmilesTransformer(n_jobs=-1)

    mol_to_smiles_seq = MolToSmilesTransformer()
    mol_to_smiles_parallel = MolToSmilesTransformer(n_jobs=-1)

    mols_list_seq = mol_from_smiles_seq.transform(smiles_list)
    mols_list_parallel = mol_from_smiles_parallel.transform(smiles_list)

    smiles_list_2_seq = mol_to_smiles_seq.transform(mols_list_seq)
    smiles_list_2_parallel = mol_to_smiles_parallel.transform(mols_list_parallel)

    assert smiles_list_2_seq == smiles_list
    assert smiles_list_2_seq == smiles_list_2_parallel


def test_from_invalid_smiles(smiles_list):
    invalid_smiles_list = ["[H]=[H]", "invalid"]
    mol_from_smiles = MolFromSmilesTransformer(valid_only=False)
    mols_list = mol_from_smiles.transform(smiles_list + invalid_smiles_list)

    mol_from_smiles = MolFromSmilesTransformer(valid_only=True)
    mols_list_2 = mol_from_smiles.transform(smiles_list + invalid_smiles_list)

    assert len(mols_list) == len(smiles_list) + len(invalid_smiles_list)
    assert len(mols_list_2) == len(smiles_list)
