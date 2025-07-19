import numpy as np
import pytest

from skfp.filters import REOSFilter


@pytest.fixture
def smiles_passing_reos() -> list[str]:
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
def smiles_failing_reos() -> list[str]:
    return [
        # Rifampin
        r"C[C@H]1/C=C/C=C(\C(=O)NC2=C(C(=C3C(=C2O)C(=C(C4=C3C(=O)[C@](O4)(O/C=C/[C@@H]([C@H]([C@H]([C@@H]([C@@H]([C@@H]([C@H]1O)C)O)C)OC(=O)C)C)OC)C)C)O)O)/C=N/N5CCN(CC5)C)/C",
        "CCO",  # Ethanol
    ]


def test_mols_passing_reos(smiles_passing_reos):
    mol_filter = REOSFilter()
    smiles_filtered = mol_filter.transform(smiles_passing_reos)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_reos)


def test_mols_partially_passing_reos(smiles_passing_one_fail):
    mol_filter = REOSFilter(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(smiles_passing_one_fail)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_one_fail)


def test_mols_failing_reos(smiles_failing_reos):
    mol_filter = REOSFilter()
    smiles_filtered = mol_filter.transform(smiles_failing_reos)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_reos_filter_return_indicators(
    smiles_passing_reos,
    smiles_failing_reos,
    smiles_passing_one_fail,
):
    all_smiles = smiles_passing_reos + smiles_failing_reos + smiles_passing_one_fail

    mol_filter = REOSFilter(return_type="indicators")
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_reos)
        + [False] * len(smiles_failing_reos)
        + [False] * len(smiles_passing_one_fail),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)

    mol_filter = REOSFilter(allow_one_violation=True, return_type="indicators")
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_reos)
        + [False] * len(smiles_failing_reos)
        + [True] * len(smiles_passing_one_fail),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)


def test_reos_filter_parallel(smiles_list):
    mol_filter = REOSFilter()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = REOSFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel


def test_reos_transform_x_y(smiles_passing_reos, smiles_failing_reos):
    all_smiles = smiles_passing_reos + smiles_failing_reos
    labels = np.array([1] * len(smiles_passing_reos) + [0] * len(smiles_failing_reos))

    filt = REOSFilter()
    mols, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert len(mols) == len(smiles_passing_reos)
    assert np.all(labels_filt == 1)

    filt = REOSFilter(return_type="indicators")
    indicators, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert np.sum(indicators) == len(smiles_passing_reos)
    assert np.array_equal(indicators, labels_filt)


def test_reos_condition_names():
    filt = REOSFilter()
    condition_names = filt.get_feature_names_out()

    assert isinstance(condition_names, np.ndarray)
    assert condition_names.shape == (7,)


def test_reos_return_condition_indicators(smiles_passing_reos, smiles_failing_reos):
    all_smiles = smiles_passing_reos + smiles_failing_reos

    filt = REOSFilter(return_type="condition_indicators")
    condition_indicators = filt.transform(all_smiles)

    assert isinstance(condition_indicators, np.ndarray)
    assert condition_indicators.shape == (len(all_smiles), 7)
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))


def test_reos_return_condition_indicators_transform_x_y(
    smiles_passing_reos, smiles_failing_reos
):
    all_smiles = smiles_passing_reos + smiles_failing_reos
    labels = np.array([1] * len(smiles_passing_reos) + [0] * len(smiles_failing_reos))

    filt = REOSFilter(return_type="condition_indicators")
    condition_indicators, y = filt.transform_x_y(all_smiles, labels)

    assert isinstance(condition_indicators, np.ndarray)
    assert condition_indicators.shape == (len(all_smiles), 7)
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))
    assert len(condition_indicators) == len(y)
