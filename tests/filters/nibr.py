import numpy as np
import pytest
from rdkit.Chem import Mol

from skfp.filters import NIBRFilter


@pytest.fixture
def smiles_passing_nibr() -> list[str]:
    return [
        # paracetamol
        "CC(=O)Nc1ccc(O)cc1",
        # caffeine
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        # nicotine
        "c1ncccc1[C@@H]2CCCN2C",
    ]


@pytest.fixture
def smiles_failing_nibr() -> list[str]:
    # taken from:
    # https://github.com/rdkit/rdkit/blob/master/Contrib/NIBRSubstructureFilters/SubstructureFilter_HitTriaging_wPubChemExamples.csv
    return [
        "CCC12CCCC3=C1C(=C(S3)N)C(=O)NC2=O",
        "CCOC(=O)N=NN(C)N=NC(=O)OCC",
        "CC1=CN(C=N1)C2=C(C=C(C=C2)C(=O)C3=C(C=NN(C3=O)C(C)C4=CC=C(C=C4)F)Br)OC",
    ]


def test_mols_passing_nibr(smiles_passing_nibr):
    filt = NIBRFilter()
    smiles_filtered = filt.transform(smiles_passing_nibr)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_nibr)


def test_mols_failing_nibr(smiles_failing_nibr):
    filt = NIBRFilter()
    smiles_filtered = filt.transform(smiles_failing_nibr)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_nibr_smiles_and_mol_input(smiles_list, mols_list):
    filt = NIBRFilter()
    smiles_filtered = filt.transform(smiles_list)
    mols_filtered = filt.transform(mols_list)

    assert all(isinstance(x, str) for x in smiles_filtered)
    assert all(isinstance(x, Mol) for x in mols_filtered)
    assert len(smiles_filtered) == len(mols_filtered)


def test_nibr_parallel(smiles_list):
    filt = NIBRFilter()
    mols_filtered_sequential = filt.transform(smiles_list)

    filt = NIBRFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = filt.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel


def test_mols_loose_nibr(mols_list):
    filt = NIBRFilter()
    mols_filtered_nibr = filt.transform(mols_list)

    filt = NIBRFilter(allow_one_violation=True)
    mols_filtered_loose_nibr = filt.transform(mols_list)

    assert len(mols_filtered_nibr) <= len(mols_filtered_loose_nibr)


def test_nibr_return_indicators(smiles_passing_nibr, smiles_failing_nibr):
    all_smiles = smiles_passing_nibr + smiles_failing_nibr

    filt = NIBRFilter(return_indicators=True)
    filter_indicators = filt.transform(all_smiles)

    assert len(filter_indicators) == len(all_smiles)
    assert isinstance(filter_indicators, np.ndarray)
    assert np.issubdtype(filter_indicators.dtype, bool)
    assert np.all(np.isin(filter_indicators, [0, 1]))


def test_nibr_transform_x_y(smiles_passing_nibr, smiles_failing_nibr):
    all_smiles = smiles_passing_nibr + smiles_failing_nibr
    labels = np.array([1] * len(smiles_passing_nibr) + [0] * len(smiles_failing_nibr))

    filt = NIBRFilter()
    mols, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert len(mols) == len(smiles_passing_nibr)
    assert np.all(labels_filt == 1)

    filt = NIBRFilter(return_indicators=True)
    indicators, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert np.sum(indicators) == len(smiles_passing_nibr)
    assert np.array_equal(indicators, labels_filt)
