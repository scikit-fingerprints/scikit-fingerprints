# flake8: noqa E501

import numpy as np
import pytest
from rdkit.Chem import Mol, MolFromSmiles

from skfp.preprocessing import RuleOfDrugLikeSoft


@pytest.fixture
def smiles_passing_rule_of_druglike_soft() -> list[str]:
    return ["CC1=CC2=C(C=C1C)N(C3=NC(=O)NC(=O)C3=N2)C[C@@H]([C@@H]([C@@H](CO)O)O)O"]


@pytest.fixture
def smiles_passing_one_fail() -> list[str]:
    return [
        r"CC(=O)OCC1=C(N2[C@@H]([C@@H](C2=O)NC(=O)/C(=N\OC)/C3=CSC(=N3)N)SC1)C(=O)O",  # Cefotaxime
        "CN1CC[C@@]23CCCC[C@@H]2[C@@H]1CC4=C3C=C(C=C4)OC",  # Dextromethorphan
    ]


@pytest.fixture
def smiles_failing_rule_of_druglike_soft() -> list[str]:
    return [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CN1CC[C@@]23CCCC[C@@H]2[C@@H]1CC4=C3C=C(C=C4)OC",  # Dextromethorphan
    ]


def test_mols_passing_rule_of_druglike_soft(smiles_passing_rule_of_druglike_soft):
    mol_filter = RuleOfDrugLikeSoft()
    smiles_filtered = mol_filter.transform(smiles_passing_rule_of_druglike_soft)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_rule_of_druglike_soft)


def test_mols_partially_passing_rule_of_druglike_soft(smiles_passing_one_fail):
    mol_filter = RuleOfDrugLikeSoft(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(smiles_passing_one_fail)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_one_fail)


def test_mols_failling_rule_of_druglike_soft(smiles_failing_rule_of_druglike_soft):
    mol_filter = RuleOfDrugLikeSoft()
    smiles_filtered = mol_filter.transform(smiles_failing_rule_of_druglike_soft)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_rule_of_druglike_soft_return_indicators(
    smiles_passing_rule_of_druglike_soft,
    smiles_failing_rule_of_druglike_soft,
    smiles_passing_one_fail,
):
    all_smiles = (
        smiles_passing_rule_of_druglike_soft
        + smiles_failing_rule_of_druglike_soft
        + smiles_passing_one_fail
    )

    mol_filter = RuleOfDrugLikeSoft(return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_rule_of_druglike_soft)
        + [False] * len(smiles_failing_rule_of_druglike_soft)
        + [False] * len(smiles_passing_one_fail),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)

    mol_filter = RuleOfDrugLikeSoft(allow_one_violation=True, return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_rule_of_druglike_soft)
        + [False] * len(smiles_failing_rule_of_druglike_soft)
        + [True] * len(smiles_passing_one_fail),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)


def test_rule_of_druglike_soft_parallel(smiles_list):
    mol_filter = RuleOfDrugLikeSoft()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = RuleOfDrugLikeSoft(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel
