from typing import List

import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from rdkit.Chem import Mol, MolFromSmiles


def pytest_addoption(parser) -> None:
    parser.addoption("--num_mols", action="store", default="100")


@pytest.fixture(scope="session")
def smiles_list(request: FixtureRequest) -> List[str]:
    # handle different paths, e.g. from CLI and PyCharm
    try:
        smiles = _load_smiles("hiv_mol.csv.zip")
    except Exception:
        smiles = _load_smiles("tests/hiv_mol.csv.zip")

    return smiles[: int(request.config.getoption("--num_mols"))]


@pytest.fixture(scope="session")
def mols_list(smiles_list) -> List[Mol]:
    return [MolFromSmiles(smi) for smi in smiles_list]


@pytest.fixture(scope="session")
def smallest_smiles_list(request: FixtureRequest) -> List[str]:
    """
    Returns shortest SMILES, i.e. for smallest molecules, for use with
    computationally demanding fingerprints.
    """

    # handle different paths, e.g. from CLI and PyCharm
    try:
        smiles = _load_smiles("hiv_mol.csv.zip")
    except Exception:
        smiles = _load_smiles("tests/hiv_mol.csv.zip")

    smiles.sort(key=len)
    return smiles[: int(request.config.getoption("--num_mols"))]


@pytest.fixture(scope="session")
def smallest_mols_list(smallest_smiles_list) -> List[Mol]:
    """
    Returns shortest molecules, for use with computationally demanding fingerprints.
    """
    return [MolFromSmiles(smi) for smi in smallest_smiles_list]


def _load_smiles(file_path: str) -> List[str]:
    return pd.read_csv(file_path)["smiles"].tolist()
