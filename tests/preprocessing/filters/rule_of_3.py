import numpy as np
import pytest

from skfp.preprocessing import RuleOf3


@pytest.fixture
def smiles_passing_basic_rule_of_3() -> list[str]:
    return [
        "C=CCNC(=S)NCc1ccccc1OC",
        "C=CCOc1ccc(Br)cc1/C=N/O",
        "c1ccc(-c2nnc3n2CCCC3)cc1",
    ]


@pytest.fixture
def smiles_failing_basic_rule_of_3() -> list[str]:
    return [
        "C=CCNC(=O)CSCc1ccc(Br)cc1",
        "C=CCNc1nc(C)nc(OC)n1",
        "C=CCOc1ccc2ccccc2c1/C=N/O",
    ]


@pytest.fixture
def smiles_passing_one_violation_basic_rule_of_3() -> list[str]:
    return [
        "c1ccc(CCNc2ccc3nncn3n2)cc1",
        "c1ccc(CCNc2ncnc3nc[nH]c23)cc1",
        "c1ccc(Cn2nnc3cnccc32)cc1",
    ]


@pytest.fixture
def smiles_passing_extended_rule_of_3() -> list[str]:
    return ["C=CCNC(C)=C1C(=O)CCC1=O", "C=CCNc1ncnc2ccccc12", "C=CCSc1nc2ccccc2[nH]1"]


@pytest.fixture
def smiles_failing_extended_rule_of_3() -> list[str]:
    return [
        "c1ccc(Cn2nnnc2C(c2ccccn2)N2CCN(c3ccccc3)CC2)cc1",
        "c1ccc(Cn2nnnc2CN2CCN(c3ncccn3)CC2)cc1",
        "c1ccc(CNc2nc(NCCN3CCOCC3)nc(N3CCOCC3)n2)cc1",
    ]


@pytest.fixture
def smiles_passing_one_violation_extended_rule_of_3() -> list[str]:
    return ["C=CCOc1ccc(Cl)cc1C(=O)O", "C=CCSc1ncccc1C(=O)O", "c1ccc(CCCNc2ncccn2)cc1"]


def test_mols_passing_basic_rule_of_3(smiles_passing_basic_rule_of_3):
    mol_filter = RuleOf3()
    smiles_filtered = mol_filter.transform(smiles_passing_basic_rule_of_3)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_basic_rule_of_3)


def test_mols_passing_extended_rule_of_3(smiles_passing_extended_rule_of_3):
    mol_filter = RuleOf3(extended=True)
    smiles_filtered = mol_filter.transform(smiles_passing_extended_rule_of_3)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_extended_rule_of_3)


def test_mols_failing_basic_rule_of_3(smiles_failing_basic_rule_of_3):
    mol_filter = RuleOf3()
    smiles_filtered = mol_filter.transform(smiles_failing_basic_rule_of_3)
    assert all(isinstance(x, str) for x in smiles_failing_basic_rule_of_3)
    assert len(smiles_filtered) == 0


def test_mols_failing_extended_rule_of_3(smiles_failing_extended_rule_of_3):
    mol_filter = RuleOf3(extended=True)
    smiles_filtered = mol_filter.transform(smiles_failing_extended_rule_of_3)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_mols_passing_with_violation_basic_rule_of_3(
    smiles_passing_one_violation_basic_rule_of_3,
):
    mol_filter = RuleOf3(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(smiles_passing_one_violation_basic_rule_of_3)
    assert len(smiles_filtered) == 3

    mol_filter = RuleOf3(allow_one_violation=False)
    smiles_filtered = mol_filter.transform(smiles_passing_one_violation_basic_rule_of_3)
    assert len(smiles_filtered) == 0


def test_mols_passing_with_violation_extended_rule_of_3(
    smiles_passing_one_violation_extended_rule_of_3,
):
    mol_filter = RuleOf3(allow_one_violation=True, extended=True)
    smiles_filtered = mol_filter.transform(
        smiles_passing_one_violation_extended_rule_of_3
    )
    assert len(smiles_filtered) == 3

    mol_filter = RuleOf3(allow_one_violation=False, extended=True)
    smiles_filtered = mol_filter.transform(
        smiles_passing_one_violation_extended_rule_of_3
    )
    assert len(smiles_filtered) == 0


def test_rule_of_3_return_indicators(
    smiles_passing_extended_rule_of_3,
    smiles_failing_extended_rule_of_3,
    smiles_passing_one_violation_extended_rule_of_3,
):
    all_smiles = (
        smiles_passing_extended_rule_of_3
        + smiles_failing_extended_rule_of_3
        + smiles_passing_one_violation_extended_rule_of_3
    )

    mol_filter = RuleOf3(extended=True, return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_extended_rule_of_3)
        + [False] * len(smiles_failing_extended_rule_of_3)
        + [False] * len(smiles_passing_one_violation_extended_rule_of_3),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)

    mol_filter = RuleOf3(
        extended=True, allow_one_violation=True, return_indicators=True
    )
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_extended_rule_of_3)
        + [False] * len(smiles_failing_extended_rule_of_3)
        + [True] * len(smiles_passing_one_violation_extended_rule_of_3),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)


def test_rule_of_3_parallel(smiles_list):
    mol_filter = RuleOf3(extended=True)
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = RuleOf3(extended=True, n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel
