import numpy as np
import pytest

from skfp.filters import PfizerFilter


@pytest.fixture
def smiles_passing_pfizer() -> list[str]:
    return [
        "COC(=O)c1ccccc1NC(=O)CSc1nc(O)c(-c2ccccc2)c(=O)[nH]1",
        "CS(=O)(=O)NCc1nnc(SCc2ccccc2C(F)(F)F)o1",
        "COCCCn1c(C)nnc1SCC(=O)NCc1ccco1",
    ]


@pytest.fixture
def smiles_failing_pfizer() -> list[str]:
    return [
        "CCOC(=O)C1=C(CN2CCc3ccccc32)NC(=O)NC1c1cc(C)ccc1C",
        "O=C(COC(=O)/C=C/c1cccc(F)c1)Nc1ccccc1",
        "Cc1ccccc1-n1c(Cc2cccs2)n[nH]c1=S",
    ]


@pytest.fixture
def smiles_passing_one_violation_pfizer() -> list[str]:
    return [
        "CCC(C)NC(=O)C1(C)CCC(=O)N1Cc1ccccc1OC",
        "Cn1c(SCC(=O)N2CCc3ccccc32)nc2cccnc21",
        "CCOC(=O)c1s/c(=N/C(=O)/C=C/c2ccccc2)n(C)c1C",
    ]


def test_mols_passing_pfizer(
    smiles_passing_pfizer,
):
    mol_filter = PfizerFilter()
    smiles_filtered = mol_filter.transform(smiles_passing_pfizer)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_pfizer)


def test_mols_failing_pfizer(smiles_failing_pfizer):
    mol_filter = PfizerFilter()
    smiles_filtered = mol_filter.transform(smiles_failing_pfizer)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_mols_passing_with_violation_pfizer(
    smiles_passing_one_violation_pfizer,
):
    mol_filter = PfizerFilter(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(smiles_passing_one_violation_pfizer)
    assert len(smiles_filtered) == 3

    mol_filter = PfizerFilter(allow_one_violation=False)
    smiles_filtered = mol_filter.transform(smiles_passing_one_violation_pfizer)
    assert len(smiles_filtered) == 0


def test_pfizer_return_indicators(
    smiles_passing_pfizer,
    smiles_failing_pfizer,
    smiles_passing_one_violation_pfizer,
):
    all_smiles = (
        smiles_passing_pfizer
        + smiles_failing_pfizer
        + smiles_passing_one_violation_pfizer
    )

    mol_filter = PfizerFilter(return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_pfizer)
        + [False] * len(smiles_failing_pfizer)
        + [False] * len(smiles_passing_one_violation_pfizer),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)

    mol_filter = PfizerFilter(allow_one_violation=True, return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_pfizer)
        + [False] * len(smiles_failing_pfizer)
        + [True] * len(smiles_passing_one_violation_pfizer),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)


def test_pfizer_parallel(smiles_list):
    mol_filter = PfizerFilter()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = PfizerFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel
