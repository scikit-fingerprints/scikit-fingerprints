from rdkit.Chem import Mol

from skfp.preprocessing import BasicZINCFilter


def test_basic_zinc(mols_list):
    pains = BasicZINCFilter()
    mols_filtered = pains.transform(mols_list)
    assert all(isinstance(x, Mol) for x in mols_filtered)
    assert len(mols_filtered) <= len(mols_list)


def test_basic_zinc_parallel(smiles_list):
    filt = BasicZINCFilter()
    smiles_filtered_sequential = filt.transform(smiles_list)

    filt = BasicZINCFilter(n_jobs=-1)
    smiles_filtered_parallel = filt.transform(smiles_list)

    assert smiles_filtered_sequential == smiles_filtered_parallel


def test_basic_zinc_allowing_one_violation(mols_list):
    filt = BasicZINCFilter()
    filt_loose = BasicZINCFilter(allow_one_violation=True)

    mols_filtered = filt.transform(mols_list)
    mols_filtered_loose = filt_loose.transform(mols_list)

    assert len(mols_filtered) <= len(mols_filtered_loose)
    assert len(mols_filtered_loose) <= len(mols_list)
