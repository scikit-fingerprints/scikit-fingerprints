from rdkit.Chem import Mol

from skfp.preprocessing import BMSFilter


def test_bms(mols_list):
    pains = BMSFilter()
    mols_filtered = pains.transform(mols_list)
    assert all(isinstance(x, Mol) for x in mols_filtered)
    assert len(mols_filtered) <= len(mols_list)


def test_bms_parallel(smiles_list):
    filt = BMSFilter()
    smiles_filtered_sequential = filt.transform(smiles_list)

    filt = BMSFilter(n_jobs=-1)
    smiles_filtered_parallel = filt.transform(smiles_list)

    assert smiles_filtered_sequential == smiles_filtered_parallel


def test_bms_allowing_one_violation(mols_list):
    filt = BMSFilter()
    filt_loose = BMSFilter(allow_one_violation=True)

    mols_filtered = filt.transform(mols_list)
    mols_filtered_loose = filt_loose.transform(mols_list)

    assert len(mols_filtered) <= len(mols_filtered_loose)
    assert len(mols_filtered_loose) <= len(mols_list)
