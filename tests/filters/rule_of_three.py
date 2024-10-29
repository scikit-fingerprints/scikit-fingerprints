import numpy as np
import pytest

from skfp.filters import RuleOfThreeFilter


@pytest.fixture
def smiles_passing_basic_rule_of_three() -> list[str]:
    return [
        "C=CCNC(=S)NCc1ccccc1OC",
        "C=CCOc1ccc(Br)cc1/C=N/O",
        "c1ccc(-c2nnc3n2CCCC3)cc1",
    ]


@pytest.fixture
def smiles_failing_basic_rule_of_three() -> list[str]:
    return [
        "C=CCNC(=O)CSCc1ccc(Br)cc1",
        "C=CCNc1nc(C)nc(OC)n1",
        "C=CCOc1ccc2ccccc2c1/C=N/O",
    ]


@pytest.fixture
def smiles_passing_one_violation_basic_rule_of_three() -> list[str]:
    return [
        "c1ccc(CCNc2ccc3nncn3n2)cc1",
        "c1ccc(CCNc2ncnc3nc[nH]c23)cc1",
        "c1ccc(Cn2nnc3cnccc32)cc1",
    ]


@pytest.fixture
def smiles_passing_extended_rule_of_three() -> list[str]:
    return ["C=CCNC(C)=C1C(=O)CCC1=O", "C=CCNc1ncnc2ccccc12", "C=CCSc1nc2ccccc2[nH]1"]


@pytest.fixture
def smiles_failing_extended_rule_of_three() -> list[str]:
    return [
        "c1ccc(Cn2nnnc2C(c2ccccn2)N2CCN(c3ccccc3)CC2)cc1",
        "c1ccc(Cn2nnnc2CN2CCN(c3ncccn3)CC2)cc1",
        "c1ccc(CNc2nc(NCCN3CCOCC3)nc(N3CCOCC3)n2)cc1",
    ]


@pytest.fixture
def smiles_passing_one_violation_extended_rule_of_three() -> list[str]:
    return ["C=CCOc1ccc(Cl)cc1C(=O)O", "C=CCSc1ncccc1C(=O)O", "c1ccc(CCCNc2ncccn2)cc1"]


def test_mols_passing_basic_rule_of_three(smiles_passing_basic_rule_of_three):
    mol_filter = RuleOfThreeFilter()
    smiles_filtered = mol_filter.transform(smiles_passing_basic_rule_of_three)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_basic_rule_of_three)


def test_mols_passing_extended_rule_of_three(smiles_passing_extended_rule_of_three):
    mol_filter = RuleOfThreeFilter(extended=True)
    smiles_filtered = mol_filter.transform(smiles_passing_extended_rule_of_three)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_extended_rule_of_three)


def test_mols_failing_basic_rule_of_three(smiles_failing_basic_rule_of_three):
    mol_filter = RuleOfThreeFilter()
    smiles_filtered = mol_filter.transform(smiles_failing_basic_rule_of_three)
    assert all(isinstance(x, str) for x in smiles_failing_basic_rule_of_three)
    assert len(smiles_filtered) == 0


def test_mols_failing_extended_rule_of_three(smiles_failing_extended_rule_of_three):
    mol_filter = RuleOfThreeFilter(extended=True)
    smiles_filtered = mol_filter.transform(smiles_failing_extended_rule_of_three)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_mols_passing_with_violation_basic_rule_of_three(
    smiles_passing_one_violation_basic_rule_of_three,
):
    mol_filter = RuleOfThreeFilter(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(
        smiles_passing_one_violation_basic_rule_of_three
    )
    assert len(smiles_filtered) == 3

    mol_filter = RuleOfThreeFilter(allow_one_violation=False)
    smiles_filtered = mol_filter.transform(
        smiles_passing_one_violation_basic_rule_of_three
    )
    assert len(smiles_filtered) == 0


def test_mols_passing_with_violation_extended_rule_of_three(
    smiles_passing_one_violation_extended_rule_of_three,
):
    mol_filter = RuleOfThreeFilter(allow_one_violation=True, extended=True)
    smiles_filtered = mol_filter.transform(
        smiles_passing_one_violation_extended_rule_of_three
    )
    assert len(smiles_filtered) == 3

    mol_filter = RuleOfThreeFilter(allow_one_violation=False, extended=True)
    smiles_filtered = mol_filter.transform(
        smiles_passing_one_violation_extended_rule_of_three
    )
    assert len(smiles_filtered) == 0


def test_rule_of_three_return_indicators(
    smiles_passing_extended_rule_of_three,
    smiles_failing_extended_rule_of_three,
    smiles_passing_one_violation_extended_rule_of_three,
):
    all_smiles = (
        smiles_passing_extended_rule_of_three
        + smiles_failing_extended_rule_of_three
        + smiles_passing_one_violation_extended_rule_of_three
    )

    mol_filter = RuleOfThreeFilter(extended=True, return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_extended_rule_of_three)
        + [False] * len(smiles_failing_extended_rule_of_three)
        + [False] * len(smiles_passing_one_violation_extended_rule_of_three),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)

    mol_filter = RuleOfThreeFilter(
        extended=True, allow_one_violation=True, return_indicators=True
    )
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_extended_rule_of_three)
        + [False] * len(smiles_failing_extended_rule_of_three)
        + [True] * len(smiles_passing_one_violation_extended_rule_of_three),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)


def test_rule_of_three_parallel(smiles_list):
    mol_filter = RuleOfThreeFilter(extended=True)
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = RuleOfThreeFilter(extended=True, n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel
