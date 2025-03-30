import os
from pathlib import Path

import huggingface_hub.constants as hf_constants
import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from rdkit.Chem import Mol, MolFromSmiles
from rdkit.Chem.PropertyMol import PropertyMol

from skfp.preprocessing import ConformerGenerator

# set longer timeouts for HuggingFace
hf_constants.DEFAULT_DOWNLOAD_TIMEOUT = 120
hf_constants.DEFAULT_ETAG_TIMEOUT = 120
hf_constants.DEFAULT_REQUEST_TIMEOUT = 120
hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = 120
hf_constants.HF_HUB_ETAG_TIMEOUT = 120


@pytest.fixture(scope="session")
def fasta_list():
    return [
        ">peptide_pm_1\nKWLRRVWRWWR\n",
        ">peptide_pm_2\nFLPAIGRVLSGIL\n",
        ">peptide_pm_3\nCGESCVWIPCISAVVGCSCKSKVCYKNGTLP\n",
        ">peptide_pm_4\nILGKLLSTAWGLLSKL\n",
        ">peptide_pm_5\nWKLFKKIPKFLHLAKKF\n",
        ">peptide_pm_6\nRAGLQFPVGRLLRRLLRRLLR\n",
        ">peptide_pm_7\nGLWSKIKTAGKSVAKAAAKAAVKAVTNAV\n",
        ">peptide_pm_8\nCGESCVYIPCLTSAIGCSCKSKVCYRNGIP\n",
    ]


def pytest_addoption(parser) -> None:
    parser.addoption("--num_mols", action="store", default="100")


@pytest.fixture(scope="session")
def smiles_list(request: FixtureRequest) -> list[str]:
    smiles = _load_test_data_smiles()
    return smiles[: int(request.config.getoption("--num_mols"))]


@pytest.fixture(scope="session")
def mols_list(smiles_list) -> list[Mol]:
    return [MolFromSmiles(smi) for smi in smiles_list]


@pytest.fixture(scope="session")
def smallest_smiles_list(request: FixtureRequest) -> list[str]:
    """
    Return shortest SMILES, i.e. for smallest molecules, for use with
    computationally demanding fingerprints.
    """
    smiles = _load_test_data_smiles()
    smiles.sort(key=len)
    return smiles[: int(request.config.getoption("--num_mols"))]


@pytest.fixture(scope="session")
def smallest_mols_list(smallest_smiles_list) -> list[Mol]:
    """
    Return smallest molecules, for use with computationally demanding fingerprints.
    """
    return [MolFromSmiles(smi) for smi in smallest_smiles_list]


@pytest.fixture(scope="session")
def mols_conformers_list(smallest_mols_list) -> list[PropertyMol]:
    conf_gen = ConformerGenerator()
    return conf_gen.transform(smallest_mols_list)


def _load_test_data_smiles() -> pd.DataFrame:
    # handle different paths and execution directories, e.g. from CLI and PyCharm
    if "tests" in os.listdir():
        df = pd.read_csv(os.path.join("tests", "hiv_mol.csv.zip"))
    elif "hiv_mol.csv.zip" in os.listdir():
        df = pd.read_csv("hiv_mol.csv.zip")
    else:
        curr_dir = Path(os.getcwd()).parent
        counter = 1
        while counter < 3:
            try:
                filepath = os.path.join(str(curr_dir), "hiv_mol.csv.zip")
                df = pd.read_csv(filepath)
                break
            except FileNotFoundError:
                curr_dir = curr_dir.parent
                counter += 1

        if counter >= 3:
            raise FileNotFoundError("File hiv_mol.csv.zip not found")

    return df["smiles"].tolist()
