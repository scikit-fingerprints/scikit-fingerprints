import numpy as np
import pytest

from skfp.filters import FAF4LeadlikeFilter


@pytest.fixture
def smiles_passing_faf4_leadlike() -> list[str]:
    return [
        # paracetamol
        "CC(=O)Nc1ccc(O)cc1",
        # Ibuprofen
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        # caffeine
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        # nicotine
        "c1ncccc1[C@@H]2CCCN2C",
    ]


@pytest.fixture
def smiles_failing_faf4_leadlike() -> list[str]:
    return [
        # Rinfampin
        r"CN1CCN(CC1)/N=C/c2c(O)c3c5C(=O)[C@@]4(C)O/C=C/[C@H](OC)[C@@H](C)[C@@H](OC(C)=O)[C@H](C)[C@H](O)[C@H](C)[C@@H](O)[C@@H](C)\C=C\C=C(\C)C(=O)Nc2c(O)c3c(O)c(C)c5O4",
        # Probucol
        "S(c1cc(c(O)c(c1)C(C)(C)C)C(C)(C)C)C(Sc2cc(c(O)c(c2)C(C)(C)C)C(C)(C)C)(C)C",
        # Kanamycin
        "O([C@@H]2[C@@H](O)[C@H](O[C@H]1O[C@H](CN)[C@@H](O)[C@H](O)[C@H]1O)[C@@H](N)C[C@H]2N)[C@H]3O[C@@H]([C@@H](O)[C@H](N)[C@H]3O)CO",
    ]


def test_mols_passing_faf4_leadlike(smiles_passing_faf4_leadlike):
    mol_filter = FAF4LeadlikeFilter()
    smiles_filtered = mol_filter.transform(smiles_passing_faf4_leadlike)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_faf4_leadlike)


def test_mols_failing_faf4_leadlike(smiles_failing_faf4_leadlike):
    mol_filter = FAF4LeadlikeFilter()
    smiles_filtered = mol_filter.transform(smiles_failing_faf4_leadlike)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_faf4_leadlike_return_indicators(
    smiles_passing_faf4_leadlike,
    smiles_failing_faf4_leadlike,
):
    all_smiles = smiles_passing_faf4_leadlike + smiles_failing_faf4_leadlike

    mol_filter = FAF4LeadlikeFilter(return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_faf4_leadlike)
        + [False] * len(smiles_failing_faf4_leadlike),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)


def test_faf4_leadlike_parallel(smiles_list):
    mol_filter = FAF4LeadlikeFilter()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = FAF4LeadlikeFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel


def test_faf4_leadlike_allow_one_violation(smiles_failing_faf4_leadlike):
    mol_filter = FAF4LeadlikeFilter()
    mols_filtered = mol_filter.transform(smiles_failing_faf4_leadlike)

    mol_filter = FAF4LeadlikeFilter(allow_one_violation=True)
    mols_filtered_one_violation = mol_filter.transform(smiles_failing_faf4_leadlike)

    assert mols_filtered == mols_filtered_one_violation


def test_faf4_leadlike_transform_x_y(
    smiles_passing_faf4_leadlike,
    smiles_failing_faf4_leadlike,
):
    all_smiles = smiles_passing_faf4_leadlike + smiles_failing_faf4_leadlike
    labels = np.array(
        [1] * len(smiles_passing_faf4_leadlike)
        + [0] * len(smiles_failing_faf4_leadlike)
    )

    filt = FAF4LeadlikeFilter()
    mols, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert len(mols) == len(smiles_passing_faf4_leadlike)
    assert np.all(labels_filt == 1)

    filt = FAF4LeadlikeFilter(return_indicators=True)
    indicators, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert np.sum(indicators) == len(smiles_passing_faf4_leadlike)
    assert np.array_equal(indicators, labels_filt)
