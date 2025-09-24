import numpy as np
import pytest
from numpy.testing import assert_equal

from skfp.filters import GSKFilter


@pytest.fixture
def smiles_passing_gsk() -> list[str]:
    return [
        "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",  # Ciprofloxacin
        "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",  # Warfarin
    ]


@pytest.fixture
def smiles_passing_one_fail() -> list[str]:
    return [
        r"CN1C(=NC(=O)C(=O)N1)SCC2=C(N3[C@@H]([C@@H](C3=O)NC(=O)/C(=N\OC)/C4=CSC(=N4)N)SC2)C(=O)O",  # Ceftriaxone
        r"CC(=O)OCC1=C(N2[C@@H]([C@@H](C2=O)NC(=O)/C(=N\OC)/C3=CSC(=N3)N)SC1)C(=O)O",  # Cefotaxime
    ]


@pytest.fixture
def smiles_failing_gsk() -> list[str]:
    return [
        # Rifampin
        r"C[C@H]1/C=C/C=C(\C(=O)NC2=C(C(=C3C(=C2O)C(=C(C4=C3C(=O)[C@](O4)(O/C=C/[C@@H]([C@H]([C@H]([C@@H]([C@@H]([C@@H]([C@H]1O)C)O)C)OC(=O)C)C)OC)C)C)O)O)/C=N/N5CCN(CC5)C)/C",
        "O=C(O)c1ccccc1c2ccc(cc2)Cn3c4cc(cc(c4nc3CCC)C)c5nc6ccccc6n5C",  # Telmisartan
    ]


def test_mols_passing_gsk(smiles_passing_gsk):
    mol_filter = GSKFilter()
    smiles_filtered = mol_filter.transform(smiles_passing_gsk)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert_equal(len(smiles_filtered), len(smiles_passing_gsk))


def test_mols_partially_passing_gsk(smiles_passing_one_fail):
    mol_filter = GSKFilter(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(smiles_passing_one_fail)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert_equal(len(smiles_filtered), len(smiles_passing_one_fail))


def test_mols_failing_gsk(smiles_failing_gsk):
    mol_filter = GSKFilter()
    smiles_filtered = mol_filter.transform(smiles_failing_gsk)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert_equal(len(smiles_filtered), 0)


def test_gsk_return_indicators(
    smiles_passing_gsk,
    smiles_failing_gsk,
    smiles_passing_one_fail,
):
    all_smiles = smiles_passing_gsk + smiles_failing_gsk + smiles_passing_one_fail

    mol_filter = GSKFilter(return_type="indicators")
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_gsk)
        + [False] * len(smiles_failing_gsk)
        + [False] * len(smiles_passing_one_fail),
        dtype=bool,
    )
    assert_equal(filter_indicators, expected_indicators)

    mol_filter = GSKFilter(allow_one_violation=True, return_type="indicators")
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_gsk)
        + [False] * len(smiles_failing_gsk)
        + [True] * len(smiles_passing_one_fail),
        dtype=bool,
    )
    assert_equal(filter_indicators, expected_indicators)


def test_gsk_parallel(smiles_list):
    mol_filter = GSKFilter()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = GSKFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert_equal(mols_filtered_sequential, mols_filtered_parallel)


def test_gsk_transform_x_y(smiles_passing_gsk, smiles_failing_gsk):
    all_smiles = smiles_passing_gsk + smiles_failing_gsk
    labels = np.array([1] * len(smiles_passing_gsk) + [0] * len(smiles_failing_gsk))

    filt = GSKFilter()
    mols, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert_equal(len(mols), len(smiles_passing_gsk))
    assert np.all(labels_filt == 1)

    filt = GSKFilter(return_type="indicators")
    indicators, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert_equal(np.sum(indicators), len(smiles_passing_gsk))
    assert_equal(indicators, labels_filt)


def test_gsk_condition_names():
    filt = GSKFilter()
    condition_names = filt.get_feature_names_out()

    assert isinstance(condition_names, np.ndarray)
    assert_equal(condition_names.shape, (2,))


def test_gsk_return_condition_indicators(smiles_passing_gsk, smiles_failing_gsk):
    all_smiles = smiles_passing_gsk + smiles_failing_gsk

    filt = GSKFilter(return_type="condition_indicators")
    condition_indicators = filt.transform(all_smiles)

    assert isinstance(condition_indicators, np.ndarray)
    assert_equal(condition_indicators.shape, (len(all_smiles), 2))
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))


def test_gsk_return_condition_indicators_transform_x_y(
    smiles_passing_gsk, smiles_failing_gsk
):
    all_smiles = smiles_passing_gsk + smiles_failing_gsk
    labels = np.array([1] * len(smiles_passing_gsk) + [0] * len(smiles_failing_gsk))

    filt = GSKFilter(return_type="condition_indicators")
    condition_indicators, y = filt.transform_x_y(all_smiles, labels)

    assert isinstance(condition_indicators, np.ndarray)
    assert_equal(condition_indicators.shape, (len(all_smiles), 2))
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))
    assert_equal(len(condition_indicators), len(y))
