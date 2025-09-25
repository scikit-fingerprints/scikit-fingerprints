import os

import pytest
from numpy.testing import assert_equal
from rdkit.Chem import Mol

from skfp.preprocessing import MolFromSDFTransformer, MolToSDFTransformer


@pytest.fixture
def sdf_in_file_path():
    # L-alanine
    # https://www.molinstincts.com/sdf-mol-file/L-alanine-sdf-CT1000647025.html
    return _get_sdf_file_path("mol_in.sdf")


@pytest.fixture
def sdf_out_file_path():
    return _get_sdf_file_path("mol_out.sdf")


def test_mol_from_sdf(sdf_in_file_path):
    mol_from_sdf = MolFromSDFTransformer()
    mols = mol_from_sdf.transform(sdf_in_file_path)

    assert_equal(len(mols), 1)
    assert all(isinstance(x, Mol) for x in mols)


def test_mol_to_sdf(mols_list, sdf_out_file_path):
    mol_to_sdf = MolToSDFTransformer(sdf_out_file_path)
    mol_to_sdf.transform(mols_list)

    assert os.path.exists(sdf_out_file_path)


def test_mol_to_and_from_sdf(mols_list, sdf_out_file_path):
    mol_from_sdf = MolFromSDFTransformer()
    mol_to_sdf = MolToSDFTransformer(sdf_out_file_path)

    mol_to_sdf.transform(mols_list)
    mols_list_2 = mol_from_sdf.transform(sdf_out_file_path)

    assert_equal(len(mols_list_2), len(mols_list))
    assert all(isinstance(x, Mol) for x in mols_list_2)


def test_error_nonexistent_sdf_file():
    mol_from_sdf = MolFromSDFTransformer()
    with pytest.raises(FileNotFoundError):
        mol_from_sdf.transform("nonexistent.sdf")


def _get_sdf_file_path(filename: str) -> str:
    if "tests" in os.listdir():
        return os.path.join("tests", "preprocessing", "input_output", "data", filename)
    if "preprocessing" in os.listdir():
        return os.path.join("preprocessing", "input_output", "data", filename)
    if "input_output" in os.listdir():
        return os.path.join("input_output", "data", filename)
    raise FileNotFoundError(f"File {filename} not found")
