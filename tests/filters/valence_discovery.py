import numpy as np
import pytest

from skfp.filters import ValenceDiscoveryFilter


@pytest.fixture
def smiles_passing_valence_discovery() -> list[str]:
    return [
        # chlordiazepoxide
        "ClC1=CC2=C(N=C(NC)C[N+]([O-])=C2C3=CC=CC=C3)C=C1",
        # cortisol
        r"O=C4\C=C2/[C@]([C@H]1[C@@H](O)C[C@@]3([C@@](O)(C(=O)CO)CC[C@H]3[C@@H]1CC2)C)(C)CC4",
    ]


@pytest.fixture
def smiles_failing_valence_discovery() -> list[str]:
    return [
        # pyrene
        "c1cc2cccc3c2c4c1cccc4cc3",
        # disodium tetracarbonylferrate
        "[Na+].[Na+].[O+]#[C][Fe-6]([C]#[O+])([C]#[O+])[C]#[O+]",
    ]


@pytest.fixture
def smiles_passing_one_violation_valence_discovery() -> list[str]:
    return [
        # paracetamol
        "CC(=O)Nc1ccc(O)cc1",
        # Ibuprofen
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        # caffeine
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        # testosterone
        r"O=C4\C=C2/[C@]([C@H]1CC[C@@]3([C@@H](O)CC[C@H]3[C@@H]1CC2)C)(C)CC4",
    ]


def test_mols_passing_valence_discovery(smiles_passing_valence_discovery):
    mol_filter = ValenceDiscoveryFilter()
    smiles_filtered = mol_filter.transform(smiles_passing_valence_discovery)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_valence_discovery)


def test_mols_partially_passing_valence_discovery(
    smiles_passing_one_violation_valence_discovery,
):
    mol_filter = ValenceDiscoveryFilter(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(
        smiles_passing_one_violation_valence_discovery
    )
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_one_violation_valence_discovery)


def test_mols_failing_valence_discovery(smiles_failing_valence_discovery):
    mol_filter = ValenceDiscoveryFilter()
    smiles_filtered = mol_filter.transform(smiles_failing_valence_discovery)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_valence_discovery_return_indicators(
    smiles_passing_valence_discovery,
    smiles_failing_valence_discovery,
    smiles_passing_one_violation_valence_discovery,
):
    all_smiles = (
        smiles_passing_valence_discovery
        + smiles_failing_valence_discovery
        + smiles_passing_one_violation_valence_discovery
    )

    mol_filter = ValenceDiscoveryFilter(return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_valence_discovery)
        + [False] * len(smiles_failing_valence_discovery)
        + [False] * len(smiles_passing_one_violation_valence_discovery),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)

    mol_filter = ValenceDiscoveryFilter(
        allow_one_violation=True, return_indicators=True
    )
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_valence_discovery)
        + [False] * len(smiles_failing_valence_discovery)
        + [True] * len(smiles_passing_one_violation_valence_discovery),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)


def test_valence_discovery_parallel(smiles_list):
    mol_filter = ValenceDiscoveryFilter()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = ValenceDiscoveryFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel
