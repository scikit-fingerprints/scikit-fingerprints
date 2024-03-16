import pandas as pd
import pytest
from preprocessing import ConformerGenerator
from rdkit.Chem import Mol, MolFromSmiles


@pytest.fixture(scope="session")
def smiles_list() -> list[str]:
    try:
        smiles = pd.read_csv("hiv_mol.csv.zip")["smiles"]
    except Exception:
        smiles = pd.read_csv("tests/hiv_mol.csv.zip")["smiles"]
    return smiles.tolist()[:100]


@pytest.fixture(scope="session")
def mols_list(smiles_list) -> list[Mol]:
    return [MolFromSmiles(smi) for smi in smiles_list]


@pytest.fixture(scope="session")
def smallest_smiles_list() -> list[str]:
    # list of shortest SMILES, for computationally demanding fingerprints
    try:
        smiles = pd.read_csv("hiv_mol.csv.zip")["smiles"]
    except Exception:
        smiles = pd.read_csv("tests/hiv_mol.csv.zip")["smiles"]

    smiles = smiles.sort_values(key=lambda smi: smi.str.len())
    return smiles.tolist()[:100]


@pytest.fixture(scope="session")
def smallest_mols_list(smallest_smiles_list) -> list[Mol]:
    # list of smallest molecules, for computationally demanding fingerprints
    return [MolFromSmiles(smi) for smi in smallest_smiles_list]


@pytest.fixture(scope="session")
def mols_conformers_list(smallest_mols_list) -> list[Mol]:
    conf_gen = ConformerGenerator()
    return conf_gen.transform(smallest_mols_list)
