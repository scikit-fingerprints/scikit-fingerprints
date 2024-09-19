import numpy as np
import pytest

from skfp.preprocessing import RuleOfFour


@pytest.fixture
def smiles_passing_rule_of_four() -> list[str]:
    return [
        "c1ccc2oc(-c3ccc(Nc4nc(N5CCCCC5)nc(N5CCOCC5)n4)cc3)nc2c1",
        "c1csc(-c2csc3nc(CN4CCOCC4)nc(NCc4ccc5c(c4)OCO5)c23)c1",
        "CC(=O)C1=C(C)NC(SCC(=O)c2ccc(-c3ccccc3)cc2)=C(C#N)C1c1cccnc1",
    ]


@pytest.fixture
def smiles_failing_rule_of_four() -> list[str]:
    return [
        "c1ccc2c(c1)nc1n2C2(N3CCOCC3)CCCC2C1N1CCOCC1",
        "c1ccc2c(c1)oc1c(N3CCN(Cc4ccc5c(c4)OCO5)CC3)ncnc12",
        "CC(=O)C1=C(O)C(=O)N(c2cccc(C(=O)O)c2)C1c1ccccc1F",
    ]


@pytest.fixture
def smiles_passing_one_violation_rule_of_four() -> list[str]:
    return [
        "c1nc(N2CCOCC2)c2sc3nc(N4CCOCC4)c4c(c3c2n1)CCCC4",
        r"CC(/C=C1\SC(=S)N(C(Cc2ccccc2)C(=O)O)C1=O)=C\c1ccccc1",
        r"CC(=O)/N=C1\SC2CS(=O)(=O)CC2N1c1ccc(-c2nc3ccc(C)cc3s2)cc1",
    ]


def test_mols_passing_rule_of_four(
    smiles_passing_rule_of_four,
):
    mol_filter = RuleOfFour()
    smiles_filtered = mol_filter.transform(smiles_passing_rule_of_four)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_rule_of_four)


def test_mols_failing_rule_of_four(smiles_failing_rule_of_four):
    mol_filter = RuleOfFour()
    smiles_filtered = mol_filter.transform(smiles_failing_rule_of_four)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_mols_passing_with_violation_rule_of_four(
    smiles_passing_one_violation_rule_of_four,
):
    mol_filter = RuleOfFour(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(smiles_passing_one_violation_rule_of_four)
    assert len(smiles_filtered) == 3

    mol_filter = RuleOfFour(allow_one_violation=False)
    smiles_filtered = mol_filter.transform(smiles_passing_one_violation_rule_of_four)
    assert len(smiles_filtered) == 0


def test_rule_of_four_return_indicators(
    smiles_passing_rule_of_four,
    smiles_failing_rule_of_four,
    smiles_passing_one_violation_rule_of_four,
):
    all_smiles = (
        smiles_passing_rule_of_four
        + smiles_failing_rule_of_four
        + smiles_passing_one_violation_rule_of_four
    )

    mol_filter = RuleOfFour(return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_rule_of_four)
        + [False] * len(smiles_failing_rule_of_four)
        + [False] * len(smiles_passing_one_violation_rule_of_four),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)

    mol_filter = RuleOfFour(allow_one_violation=True, return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_rule_of_four)
        + [False] * len(smiles_failing_rule_of_four)
        + [True] * len(smiles_passing_one_violation_rule_of_four),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)


def test_rule_of_four_parallel(smiles_list):
    mol_filter = RuleOfFour()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = RuleOfFour(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel
