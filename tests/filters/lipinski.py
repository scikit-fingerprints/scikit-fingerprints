import numpy as np
import pytest
from rdkit.Chem import Mol

from skfp.filters import LipinskiFilter


@pytest.fixture
def smiles_passing_lipinski() -> list[str]:
    return [
        # paracetamol
        "CC(=O)Nc1ccc(O)cc1",
        # caffeine
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        # nicotine
        "c1ncccc1[C@@H]2CCCN2C",
    ]


@pytest.fixture
def smiles_failing_lipinski() -> list[str]:
    return [
        # Telaprevir
        "CCC[C@@H](C(=O)C(=O)NC1CC1)NC(=O)[C@@H]2[C@H]3CCC[C@H]"
        "3CN2C(=O)[C@H](C(C)(C)C)NC(=O)[C@H](C4CCCCC4)NC(=O)c5cnccn5",
        # Atorvastatin
        "O=C(O)C[C@H](O)C[C@H](O)CCn2c(c(c(c2c1ccc(F)cc1)c3ccccc3)C(=O)Nc4ccccc4)C(C)C",
        # Telmisartan
        "O=C(O)c1ccccc1c2ccc(cc2)Cn3c4cc(cc(c4nc3CCC)C)c5nc6ccccc6n5C",
    ]


@pytest.fixture
def smiles_one_lipinski_violation() -> list[str]:
    return [
        # Liviatin
        "CC1C2C(C3C(C(=O)C(=C(C3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)O)C(=O)N)N(C)C)O",
        # Tetracycline
        "C[C@]1(c2cccc(c2C(=O)C3=C([C@]4([C@@H](C[C@@H]31)[C@@H](C(=C(C4=O)C(=O)N)O)N(C)C)O)O)O)O",
        # Lopinavir
        "O=C(N[C@@H](Cc1ccccc1)[C@@H](O)C[C@@H](NC(=O)[C@@H](N2C(=O)NCCC2)C(C)C)Cc3ccccc3)COc4c(cccc4C)C",
    ]


def test_mols_passing_lipinski(smiles_passing_lipinski):
    filt = LipinskiFilter()
    smiles_filtered = filt.transform(smiles_passing_lipinski)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_lipinski)


def test_mols_failing_lipinski(smiles_failing_lipinski):
    filt = LipinskiFilter()
    smiles_filtered = filt.transform(smiles_failing_lipinski)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_mols_failing_strict_lipinski(smiles_one_lipinski_violation):
    filt = LipinskiFilter()
    smiles_filtered = filt.transform(smiles_one_lipinski_violation)
    assert len(smiles_filtered) == 3

    filt = LipinskiFilter(allow_one_violation=False)
    smiles_filtered = filt.transform(smiles_one_lipinski_violation)
    assert len(smiles_filtered) == 0


def test_mols_lipinski_various_conditions(
    smiles_passing_lipinski, smiles_failing_lipinski, smiles_one_lipinski_violation
):
    all_smiles = (
        smiles_passing_lipinski
        + smiles_failing_lipinski
        + smiles_one_lipinski_violation
    )

    filt = LipinskiFilter()
    mols_filtered = filt.transform(all_smiles)
    assert len(mols_filtered) == 6

    filt = LipinskiFilter(allow_one_violation=False)
    mols_filtered = filt.transform(all_smiles)
    assert len(mols_filtered) == 3


def test_lipinski_smiles_and_mol_input(smiles_list, mols_list):
    filt = LipinskiFilter()
    smiles_filtered = filt.transform(smiles_list)
    mols_filtered = filt.transform(mols_list)

    assert all(isinstance(x, str) for x in smiles_filtered)
    assert all(isinstance(x, Mol) for x in mols_filtered)
    assert len(smiles_filtered) == len(mols_filtered)


def test_lipinski_parallel(smiles_list):
    filt = LipinskiFilter()
    mols_filtered_sequential = filt.transform(smiles_list)

    filt = LipinskiFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = filt.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel


def test_lipinski_return_indicators(
    smiles_passing_lipinski, smiles_failing_lipinski, smiles_one_lipinski_violation
):
    all_smiles = (
        smiles_passing_lipinski
        + smiles_failing_lipinski
        + smiles_one_lipinski_violation
    )

    filt = LipinskiFilter(return_indicators=True)
    filter_indicators = filt.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_lipinski)
        + [False] * len(smiles_failing_lipinski)
        + [True] * len(smiles_one_lipinski_violation),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)

    filt = LipinskiFilter(allow_one_violation=False, return_indicators=True)
    filter_indicators = filt.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_lipinski)
        + [False] * len(smiles_failing_lipinski)
        + [False] * len(smiles_one_lipinski_violation),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)
