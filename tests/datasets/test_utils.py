from typing import Literal

import numpy as np
import pandas as pd
from rdkit.Chem import Mol

from skfp.preprocessing import MolFromSmilesTransformer


def run_basic_dataset_checks(
    smiles_list: list[str],
    y: np.ndarray,
    df: pd.DataFrame,
    expected_length: int,
    num_tasks: int,
    task_type: Literal[
        "binary_classification", "multiclass_classification", "regression"
    ],
) -> None:
    assert_valid_smiles_list(smiles_list, expected_length=expected_length)
    assert_valid_mols_from_smiles(smiles_list)
    assert_valid_labels(
        y,
        expected_length=expected_length,
        task_type=task_type,
        num_tasks=num_tasks,
    )
    assert_valid_dataframe(
        df,
        smiles_list,
        y,
        expected_length=expected_length,
        num_tasks=num_tasks,
    )


def assert_valid_smiles_list(smiles_list: list[str], expected_length: int) -> None:
    assert isinstance(smiles_list, list)
    assert all(isinstance(s, str) for s in smiles_list)
    assert len(smiles_list) == expected_length


def assert_valid_mols_from_smiles(smiles_list: list[str]) -> None:
    mol_from_smiles = MolFromSmilesTransformer(n_jobs=-1)
    mols = mol_from_smiles.transform(smiles_list)
    assert all(isinstance(mol, Mol) for mol in mols)
    assert all(mol.GetNumAtoms() >= 1 for mol in mols)


def assert_valid_labels(
    y: np.ndarray,
    expected_length: int,
    task_type: Literal[
        "binary_classification", "multiclass_classification", "regression"
    ],
    num_tasks: int,
) -> None:
    assert isinstance(y, np.ndarray)

    if num_tasks == 1:
        assert y.shape == (expected_length,)
    else:
        assert y.shape == (expected_length, num_tasks)

    # note that in multioutput problems we allow NaN labels
    # it results in float dtype, instead of integer
    if task_type == "binary_classification":
        validate_binary_classification_labels(y, num_tasks)
    elif task_type == "multiclass_classification":
        validate_multiclass_classification_labels(y, num_tasks)
    elif task_type == "regression":
        assert np.issubdtype(y.dtype, float)
    else:
        raise ValueError(f"task_type {task_type} not recognized")


def validate_binary_classification_labels(y: np.ndarray, num_tasks: int) -> None:
    if num_tasks == 1:
        assert np.all(np.isin(y, [0, 1]))
        assert np.issubdtype(y.dtype, np.integer)
    else:
        assert np.all(np.isin(y, [0, 1]) | np.isnan(y))
        assert np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, float)


def validate_multiclass_classification_labels(y: np.ndarray, num_tasks: int) -> None:
    if num_tasks == 1:
        assert np.all(y >= 0)
        assert np.issubdtype(y.dtype, np.integer)
    else:
        assert np.all((y >= 0) | np.isnan(y))
        assert np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, float)


def assert_valid_dataframe(
    df: pd.DataFrame,
    smiles_list: list[str],
    y: np.ndarray,
    expected_length: int,
    num_tasks: int,
) -> None:
    assert isinstance(df, pd.DataFrame)
    if "aminoseq" in df.columns:
        # SMILES + aminoseq + labels
        assert df.shape == (expected_length, num_tasks + 2)
    else:
        # SMILES + labels
        assert df.shape == (expected_length, num_tasks + 1)

    assert "SMILES" in df.columns

    df_smiles = df["SMILES"].tolist()
    df_y = df.drop(columns=["SMILES", "aminoseq"], errors="ignore").values
    if num_tasks == 1:
        df_y = df_y.ravel()

    assert df_smiles == smiles_list
    assert np.array_equal(df_y, y, equal_nan=True)
