from rdkit.Chem import Mol

from skfp.filters import MLSMRFilter


def test_mlsmr(mols_list):
    pains = MLSMRFilter()
    mols_filtered = pains.transform(mols_list)
    assert all(isinstance(x, Mol) for x in mols_filtered)
    assert len(mols_filtered) <= len(mols_list)


def test_mlsmr_parallel(smiles_list):
    filt = MLSMRFilter()
    smiles_filtered_sequential = filt.transform(smiles_list)

    filt = MLSMRFilter(n_jobs=-1)
    smiles_filtered_parallel = filt.transform(smiles_list)

    assert smiles_filtered_sequential == smiles_filtered_parallel


def test_mlsmr_allowing_one_violation(mols_list):
    filt = MLSMRFilter()
    filt_loose = MLSMRFilter(allow_one_violation=True)

    mols_filtered = filt.transform(mols_list)
    mols_filtered_loose = filt_loose.transform(mols_list)

    assert len(mols_filtered) <= len(mols_filtered_loose)
    assert len(mols_filtered_loose) <= len(mols_list)
