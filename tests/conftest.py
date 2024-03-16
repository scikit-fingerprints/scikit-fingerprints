import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from rdkit.Chem import Mol, MolFromSmiles

from skfp.preprocessing import ConformerGenerator


def pytest_addoption(parser):
    parser.addoption("--num_mols", action="store", default="100")


@pytest.fixture(scope="session")
def smiles_list(request: FixtureRequest) -> list[str]:
    # handle different paths, e.g. from CLI and PyCharm
    try:
        smiles = pd.read_csv("hiv_mol.csv.zip")["smiles"]
    except Exception:
        smiles = pd.read_csv("tests/hiv_mol.csv.zip")["smiles"]

    smiles = smiles.tolist()
    smiles = smiles[: int(request.config.getoption("--num_mols"))]
    return smiles


@pytest.fixture(scope="session")
def mols_list(smiles_list) -> list[Mol]:
    return [MolFromSmiles(smi) for smi in smiles_list]


@pytest.fixture(scope="session")
def smallest_smiles_list(request: FixtureRequest) -> list[str]:
    """
    Returns shortest SMILES, i.e. for smallest molecules, for use with
    computationally demanding fingerprints.
    """

    # handle different paths, e.g. from CLI and PyCharm
    try:
        smiles = pd.read_csv("hiv_mol.csv.zip")["smiles"]
    except Exception:
        smiles = pd.read_csv("tests/hiv_mol.csv.zip")["smiles"]

    smiles = smiles.sort_values(key=lambda smi: smi.str.len())
    smiles = smiles.tolist()
    smiles = smiles[: int(request.config.getoption("--num_mols"))]
    return smiles


@pytest.fixture(scope="session")
def smallest_mols_list(smallest_smiles_list) -> list[Mol]:
    """
    Returns shortest molecules, for use with computationally demanding fingerprints.
    """
    return [MolFromSmiles(smi) for smi in smallest_smiles_list]


@pytest.fixture(scope="session")
def mols_conformers_list(smallest_mols_list) -> list[Mol]:
    conf_gen = ConformerGenerator()
    return conf_gen.transform(smallest_mols_list)
