import numpy as np
import pytest
from numpy.testing import assert_equal

from skfp.filters import OpreaFilter


@pytest.fixture
def smiles_passing_oprea() -> list[str]:
    return [
        "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",  # Ciprofloxacin
        "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",  # Warfarin
    ]


@pytest.fixture
def smiles_passing_oprea_one_fail() -> list[str]:
    return [
        "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]


@pytest.fixture
def smiles_failing_oprea() -> list[str]:
    return [
        r"CC(=O)OCC1=C(N2[C@@H]([C@@H](C2=O)NC(=O)/C(=N\OC)/C3=CSC(=N3)N)SC1)C(=O)O",  # Cefotaxime
        r"CN1C(=NC(=O)C(=O)N1)SCC2=C(N3[C@@H]([C@@H](C3=O)NC(=O)/C(=N\OC)/C4=CSC(=N4)N)SC2)C(=O)O",  # Ceftriaxone
    ]


def test_mols_passing_oprea_filter(smiles_passing_oprea):
    mol_filter = OpreaFilter()
    smiles_filtered = mol_filter.transform(smiles_passing_oprea)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert_equal(len(smiles_filtered), len(smiles_passing_oprea))


def test_mols_partially_passing_oprea_filter(smiles_passing_oprea_one_fail):
    mol_filter = OpreaFilter(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(smiles_passing_oprea_one_fail)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert_equal(len(smiles_filtered), len(smiles_passing_oprea_one_fail))


def test_mols_failing_oprea_filter(smiles_failing_oprea):
    mol_filter = OpreaFilter()
    smiles_filtered = mol_filter.transform(smiles_failing_oprea)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert_equal(len(smiles_filtered), 0)


def test_oprea_filter_return_indicators(
    smiles_passing_oprea,
    smiles_failing_oprea,
    smiles_passing_oprea_one_fail,
):
    all_smiles = (
        smiles_passing_oprea + smiles_failing_oprea + smiles_passing_oprea_one_fail
    )

    mol_filter = OpreaFilter(return_type="indicators")
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_oprea)
        + [False] * len(smiles_failing_oprea)
        + [False] * len(smiles_passing_oprea_one_fail),
        dtype=bool,
    )
    assert_equal(filter_indicators, expected_indicators)

    mol_filter = OpreaFilter(allow_one_violation=True, return_type="indicators")
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_oprea)
        + [False] * len(smiles_failing_oprea)
        + [True] * len(smiles_passing_oprea_one_fail),
        dtype=bool,
    )
    assert_equal(filter_indicators, expected_indicators)


def test_oprea_filter_parallel(smiles_list):
    mol_filter = OpreaFilter()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = OpreaFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert_equal(mols_filtered_sequential, mols_filtered_parallel)


def test_oprea_transform_x_y(smiles_passing_oprea, smiles_failing_oprea):
    all_smiles = smiles_passing_oprea + smiles_failing_oprea
    labels = np.array([1] * len(smiles_passing_oprea) + [0] * len(smiles_failing_oprea))

    filt = OpreaFilter()
    mols, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert_equal(len(mols), len(smiles_passing_oprea))
    assert np.all(labels_filt == 1)

    filt = OpreaFilter(return_type="indicators")
    indicators, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert_equal(np.sum(indicators), len(smiles_passing_oprea))
    assert_equal(indicators, labels_filt)


def test_oprea_condition_names():
    filt = OpreaFilter()
    condition_names = filt.get_feature_names_out()

    assert isinstance(condition_names, np.ndarray)
    assert_equal(condition_names.shape, (4,))


def test_oprea_return_condition_indicators(smiles_passing_oprea, smiles_failing_oprea):
    all_smiles = smiles_passing_oprea + smiles_failing_oprea

    filt = OpreaFilter(return_type="condition_indicators")
    condition_indicators = filt.transform(all_smiles)

    assert isinstance(condition_indicators, np.ndarray)
    assert_equal(condition_indicators.shape, (len(all_smiles), 4))
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))


def test_oprea_return_condition_indicators_transform_x_y(
    smiles_passing_oprea, smiles_failing_oprea
):
    all_smiles = smiles_passing_oprea + smiles_failing_oprea
    labels = np.array([1] * len(smiles_passing_oprea) + [0] * len(smiles_failing_oprea))

    filt = OpreaFilter(return_type="condition_indicators")
    condition_indicators, y = filt.transform_x_y(all_smiles, labels)

    assert isinstance(condition_indicators, np.ndarray)
    assert_equal(condition_indicators.shape, (len(all_smiles), 4))
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))
    assert_equal(len(condition_indicators), len(y))
