import numpy as np
import pytest
from numpy.testing import assert_equal

from skfp.filters import RuleOfTwoFilter


@pytest.fixture
def smiles_passing_rule_of_two() -> list[str]:
    return ["[C-]#N", "CC=O", "C=CCc1c(C)[nH]c(N)nc1=O", "C=CCNC(=O)c1ccncc1"]


@pytest.fixture
def smiles_failing_rule_of_two() -> list[str]:
    return [
        "O=C(O)c1ccccc1c2ccc(cc2)Cn3c4cc(cc(c4nc3CCC)C)c5nc6ccccc6n5C",
        "S=C(c1ccccc1)N1CCCc2ccccc21",
        "C=CCNC(=O)CCCc1ccccc1",
    ]


@pytest.fixture
def smiles_passing_one_violation_rule_of_two() -> list[str]:
    return ["Nc1ccc(C(=O)O)c(O)c1", "C=CCC1C=C(C)CC(CC=C)N1", "C=CCc1ccccc1OCC(O)CNC"]


def test_mols_passing_rule_of_two(smiles_passing_rule_of_two):
    mol_filter = RuleOfTwoFilter()
    smiles_filtered = mol_filter.transform(smiles_passing_rule_of_two)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert_equal(len(smiles_filtered), len(smiles_passing_rule_of_two))


def test_mols_failing_rule_of_two(smiles_failing_rule_of_two):
    mol_filter = RuleOfTwoFilter()
    smiles_filtered = mol_filter.transform(smiles_failing_rule_of_two)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert_equal(len(smiles_filtered), 0)


def test_mols_passing_with_violation_rule_of_two(
    smiles_passing_one_violation_rule_of_two,
):
    mol_filter = RuleOfTwoFilter(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(smiles_passing_one_violation_rule_of_two)
    assert_equal(len(smiles_filtered), 3)

    mol_filter = RuleOfTwoFilter(allow_one_violation=False)
    smiles_filtered = mol_filter.transform(smiles_passing_one_violation_rule_of_two)
    assert_equal(len(smiles_filtered), 0)


def test_rule_of_two_return_indicators(
    smiles_passing_rule_of_two,
    smiles_failing_rule_of_two,
    smiles_passing_one_violation_rule_of_two,
):
    all_smiles = (
        smiles_passing_rule_of_two
        + smiles_failing_rule_of_two
        + smiles_passing_one_violation_rule_of_two
    )

    mol_filter = RuleOfTwoFilter(return_type="indicators")
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_rule_of_two)
        + [False] * len(smiles_failing_rule_of_two)
        + [False] * len(smiles_passing_one_violation_rule_of_two),
        dtype=bool,
    )
    assert_equal(filter_indicators, expected_indicators)

    mol_filter = RuleOfTwoFilter(allow_one_violation=True, return_type="indicators")
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_rule_of_two)
        + [False] * len(smiles_failing_rule_of_two)
        + [True] * len(smiles_passing_one_violation_rule_of_two),
        dtype=bool,
    )
    assert_equal(filter_indicators, expected_indicators)


def test_rule_of_two_parallel(smiles_list):
    mol_filter = RuleOfTwoFilter()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = RuleOfTwoFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert_equal(mols_filtered_sequential, mols_filtered_parallel)


def test_rule_of_two_transform_x_y(
    smiles_passing_rule_of_two, smiles_failing_rule_of_two
):
    all_smiles = smiles_passing_rule_of_two + smiles_failing_rule_of_two
    labels = np.array(
        [1] * len(smiles_passing_rule_of_two) + [0] * len(smiles_failing_rule_of_two)
    )

    filt = RuleOfTwoFilter()
    mols, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert_equal(len(mols), len(smiles_passing_rule_of_two))
    assert np.all(labels_filt == 1)

    filt = RuleOfTwoFilter(return_type="indicators")
    indicators, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert_equal(np.sum(indicators), len(smiles_passing_rule_of_two))
    assert_equal(indicators, labels_filt)


def test_rule_of_two_condition_names():
    filt = RuleOfTwoFilter()
    condition_names = filt.get_feature_names_out()

    assert isinstance(condition_names, np.ndarray)
    assert_equal(condition_names.shape, (4,))


def test_rule_of_two_return_condition_indicators(
    smiles_passing_rule_of_two, smiles_failing_rule_of_two
):
    all_smiles = smiles_passing_rule_of_two + smiles_failing_rule_of_two

    filt = RuleOfTwoFilter(return_type="condition_indicators")
    condition_indicators = filt.transform(all_smiles)

    assert isinstance(condition_indicators, np.ndarray)
    assert_equal(condition_indicators.shape, (len(all_smiles), 4))
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))


def test_rule_of_two_return_condition_indicators_transform_x_y(
    smiles_passing_rule_of_two, smiles_failing_rule_of_two
):
    all_smiles = smiles_passing_rule_of_two + smiles_failing_rule_of_two
    labels = np.array(
        [1] * len(smiles_passing_rule_of_two) + [0] * len(smiles_failing_rule_of_two)
    )

    filt = RuleOfTwoFilter(return_type="condition_indicators")
    condition_indicators, y = filt.transform_x_y(all_smiles, labels)

    assert isinstance(condition_indicators, np.ndarray)
    assert_equal(condition_indicators.shape, (len(all_smiles), 4))
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))
    assert_equal(len(condition_indicators), len(y))
