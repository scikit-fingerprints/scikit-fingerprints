from rdkit.Chem import Mol

from skfp.preprocessing import BrenkFilter


def test_brenk(mols_list):
    pains = BrenkFilter()
    mols_filtered = pains.transform(mols_list)
    assert all(isinstance(x, Mol) for x in mols_filtered)
    assert len(mols_filtered) <= len(mols_list)


def test_pains_parallel(smiles_list):
    filt = BrenkFilter()
    smiles_filtered_sequential = filt.transform(smiles_list)

    filt = BrenkFilter(n_jobs=-1)
    smiles_filtered_parallel = filt.transform(smiles_list)

    assert smiles_filtered_sequential == smiles_filtered_parallel


def test_brenk_allowing_one_violation(mols_list):
    filt = BrenkFilter()
    filt_loose = BrenkFilter(allow_one_violation=True)

    mols_filtered = filt.transform(mols_list)
    mols_filtered_loose = filt_loose.transform(mols_list)

    assert len(mols_filtered) <= len(mols_filtered_loose)
    assert len(mols_filtered_loose) <= len(mols_list)
