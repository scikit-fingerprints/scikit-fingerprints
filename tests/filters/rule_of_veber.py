import numpy as np
import pytest

from skfp.filters import RuleOfVeberFilter


@pytest.fixture
def smiles_passing_rule_of_veber() -> list[str]:
    return ["[C-]#N", "CC=O"]


@pytest.fixture
def smiles_passing_one_fail() -> list[str]:
    return [
        "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",  # Atorvastatin
        "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)OCC5=CC(=CC=C5)F)Cl",  # Lapatinib
    ]


@pytest.fixture
def smiles_failing_rule_of_veber() -> list[str]:
    return [
        "CC(C)(C)[C@@H](C(=O)N[C@@H](CC1=CC=CC=C1)[C@H](CN(CC2=CC=C(C=C2)C3=CC=CC=N3)NC(=O)[C@H](C(C)(C)C)NC(=O)OC)O)NC(=O)OC",  # Atazanavir
        "CC(C)[C@@H](CC1=CC(=C(C=C1)OC)OCCCOC)C[C@@H]([C@H](C[C@@H](C(C)C)C(=O)NCC(C)(C)C(=O)N)O)N",  # Aliskiren
    ]


def test_mols_passing_rule_of_veber(smiles_passing_rule_of_veber):
    mol_filter = RuleOfVeberFilter()
    smiles_filtered = mol_filter.transform(smiles_passing_rule_of_veber)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_rule_of_veber)


def test_mols_partially_passing_rule_of_veber(smiles_passing_one_fail):
    mol_filter = RuleOfVeberFilter(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(smiles_passing_one_fail)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_one_fail)


def test_mols_failing_rule_of_veber(smiles_failing_rule_of_veber):
    mol_filter = RuleOfVeberFilter()
    smiles_filtered = mol_filter.transform(smiles_failing_rule_of_veber)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_rule_of_veber_return_indicators(
    smiles_passing_rule_of_veber,
    smiles_failing_rule_of_veber,
    smiles_passing_one_fail,
):
    all_smiles = (
        smiles_passing_rule_of_veber
        + smiles_failing_rule_of_veber
        + smiles_passing_one_fail
    )

    mol_filter = RuleOfVeberFilter(return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_rule_of_veber)
        + [False] * len(smiles_failing_rule_of_veber)
        + [False] * len(smiles_passing_one_fail),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)

    mol_filter = RuleOfVeberFilter(allow_one_violation=True, return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_rule_of_veber)
        + [False] * len(smiles_failing_rule_of_veber)
        + [True] * len(smiles_passing_one_fail),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)


def test_rule_of_veber_parallel(smiles_list):
    mol_filter = RuleOfVeberFilter()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = RuleOfVeberFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel
