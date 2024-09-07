from collections.abc import Sequence
from typing import Any

from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles
from rdkit.Chem.PropertyMol import PropertyMol


def ensure_mols(X: Sequence[Any]) -> list[Mol]:
    if not all(isinstance(x, (Mol, PropertyMol, str)) for x in X):
        raise ValueError("Passed values must be either rdkit.Chem.rdChem.Mol or SMILES")

    mols = [MolFromSmiles(x) if isinstance(x, str) else x for x in X]

    if any(x is None for x in mols):
        idx = mols.index(None)
        raise ValueError(f"Could not parse {X[idx]} as molecule")

    return mols


def ensure_smiles(X: Sequence[Any]) -> list[str]:
    if not all(isinstance(x, (Mol, PropertyMol, str)) for x in X):
        raise ValueError("Passed values must be SMILES strings")
    X = [MolToSmiles(x) if isinstance(x, Mol) else x for x in X]
    return X


def require_mols_with_conf_ids(X: Sequence[Any]) -> Sequence[Mol]:
    if not all(isinstance(x, (Mol, PropertyMol)) and x.HasProp("conf_id") for x in X):
        raise ValueError(
            "Passed data must be molecules (rdkit.Chem.rdChem.Mol instances) "
            "and each must have conf_id property set. You can use "
            "ConformerGenerator to add them."
        )
    return X
