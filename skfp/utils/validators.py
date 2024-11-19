from collections.abc import Sequence
from typing import Any

from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles
from rdkit.Chem.PropertyMol import PropertyMol


def ensure_mols(X: Sequence[Any]) -> list[Mol]:
    if not all(isinstance(x, (Mol, PropertyMol, str)) for x in X):
        types = {type(x) for x in X}
        raise ValueError(
            f"Passed values must be one RDKit Mol objects or SMILES strings,"
            f"got types: {types}"
        )

    mols = [MolFromSmiles(x) if isinstance(x, str) else x for x in X]

    if any(x is None for x in mols):
        idx = mols.index(None)
        raise ValueError(f"Could not parse '{X[idx]}' at index {idx} as molecule")

    return mols


def ensure_smiles(X: Sequence[Any]) -> list[str]:
    if not all(isinstance(x, (Mol, PropertyMol, str)) for x in X):
        types = {type(x) for x in X}
        raise ValueError(f"Passed values must be SMILES strings, got types: {types}")

    X = [MolToSmiles(x) if isinstance(x, Mol) else x for x in X]
    return X


def check_strings(X: Sequence[Any]) -> None:
    for idx, x in enumerate(X):
        if not isinstance(x, str):
            raise ValueError(
                f"Passed values must be strings, got" f"type {type(x)} at index {idx}"
            )


def check_mols(X: Sequence[Any]) -> None:
    for idx, x in enumerate(X):
        if not isinstance(x, (Mol, PropertyMol)):
            raise ValueError(
                f"Passed values must be RDKit Mol objects, got type {type(x)} at index {idx}"
            )


def require_mols_with_conf_ids(X: Sequence[Any]) -> Sequence[Mol]:
    if not all(isinstance(x, (Mol, PropertyMol)) and x.HasProp("conf_id") for x in X):
        raise ValueError(
            "Passed data must be molecules (RDKit Mol objects) "
            "and each must have conf_id property set. "
            "You can use ConformerGenerator to add them."
        )
    return X
