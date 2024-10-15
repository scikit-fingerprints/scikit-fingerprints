# flake8: noqa E501

import numpy as np
import pytest

from skfp.preprocessing import RuleOfReos


@pytest.fixture
def smiles_passing_rule_of_reos() -> list[str]:
    return [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofren
        "CN1CC[C@@]23CCCC[C@@H]2[C@@H]1CC4=C3C=C(C=C4)OC",  # Dextromethorphan
    ]


@pytest.fixture
def smiles_passing_one_fail() -> list[str]:
    return [
        r"CC(=O)c1c(C(C)=O)c(C)n(CCCCn2c(C)c(C(C)=O)c(C(C)=O)c2C)c1C",
        r"CC(=O)OCC1=C(N2[C@@H]([C@@H](C2=O)NC(=O)/C(=N\OC)/C3=CSC(=N3)N)SC1)C(=O)O",  # Cefotaxime
    ]


@pytest.fixture
def smiles_failing_rule_of_reos() -> list[str]:
    return [
        r"C[C@H]1/C=C/C=C(\C(=O)NC2=C(C(=C3C(=C2O)C(=C(C4=C3C(=O)[C@](O4)(O/C=C/[C@@H]([C@H]([C@H]([C@@H]([C@@H]([C@@H]([C@H]1O)C)O)C)OC(=O)C)C)OC)C)C)O)O)/C=N/N5CCN(CC5)C)/C",  # Rifampin
        "CCO",  # Ethanol
    ]


def test_mols_passing_rule_of_reos(smiles_passing_rule_of_reos):
    mol_filter = RuleOfReos()
    smiles_filtered = mol_filter.transform(smiles_passing_rule_of_reos)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_rule_of_reos)


def test_mols_partially_passing_rule_of_reos(smiles_passing_one_fail):
    mol_filter = RuleOfReos(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(smiles_passing_one_fail)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_one_fail)


def test_mols_failing_rule_of_reos(smiles_failing_rule_of_reos):
    mol_filter = RuleOfReos()
    smiles_filtered = mol_filter.transform(smiles_failing_rule_of_reos)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_rule_of_reos_return_indicators(
    smiles_passing_rule_of_reos,
    smiles_failing_rule_of_reos,
    smiles_passing_one_fail,
):
    all_smiles = (
        smiles_passing_rule_of_reos
        + smiles_failing_rule_of_reos
        + smiles_passing_one_fail
    )

    mol_filter = RuleOfReos(return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_rule_of_reos)
        + [False] * len(smiles_failing_rule_of_reos)
        + [False] * len(smiles_passing_one_fail),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)

    mol_filter = RuleOfReos(allow_one_violation=True, return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_rule_of_reos)
        + [False] * len(smiles_failing_rule_of_reos)
        + [True] * len(smiles_passing_one_fail),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)


def test_rule_of_reos_parallel(smiles_list):
    mol_filter = RuleOfReos()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = RuleOfReos(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel
