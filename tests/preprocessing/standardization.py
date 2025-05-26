from rdkit.Chem import GetMolFrags, MolToSmiles

from skfp.preprocessing import MolStandardizer


def test_mol_standardizer(smiles_list):
    standardizer = MolStandardizer()
    standardizer_parallel = MolStandardizer(n_jobs=-1)

    mols = standardizer.transform(smiles_list)
    mols_parallel = standardizer_parallel.transform(smiles_list)

    assert len(mols) == len(mols_parallel)
    for mol, mol_2 in zip(mols, mols_parallel, strict=False):
        assert MolToSmiles(mol) == MolToSmiles(mol_2)


def test_multifragment_standardization():
    multi_fragment_smiles = [
        "[I-].[K+]",
        "N#C[S-].[K+]",
        "O=C([O-])[O-].[Ca+2]",
        "O=S(=O)([O-])[O-].O=S(=O)([O-])[O-].O=S(=O)([O-])[O-].[Fe+3].[Fe+3]",
    ]
    standardizer = MolStandardizer()
    mols = standardizer.transform(multi_fragment_smiles)
    num_frags = [len(GetMolFrags(mol)) for mol in mols]
    num_frags_expected = [2, 2, 2, 5]
    assert num_frags == num_frags_expected


def test_largest_fragment_standardization(smiles_list):
    standardizer = MolStandardizer(largest_fragment_only=True)
    mols = standardizer.transform(smiles_list)
    assert all(len(GetMolFrags(mol)) == 1 for mol in mols)

    multi_fragment_smiles = ["[I-].[K+]", "N#C[S-].[K+]", "O=C([O-])[O-].[Ca+2]"]
    mols = standardizer.transform(multi_fragment_smiles)
    assert all(len(GetMolFrags(mol)) == 1 for mol in mols)
