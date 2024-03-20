from typing import Any, Sequence

from rdkit.Chem import Mol, MolFromSmiles


def ensure_mols(X: Sequence[Any]) -> Sequence[Mol]:
    if not all(isinstance(x, (Mol, str)) for x in X):
        raise ValueError("Passed value must be either rdkit.Chem.rdChem.Mol or SMILES")

    X = [MolFromSmiles(x) if isinstance(x, str) else x for x in X]
    return X


def require_smiles(X: Sequence[Any]) -> Sequence[str]:
    if not all(isinstance(x, str) for x in X):
        raise ValueError("Passed values must be SMILES strings")
    return X


def require_mols_with_conf_ids(X: Sequence[Any]) -> Sequence[Mol]:
    if not all(isinstance(x, Mol) and hasattr(x, "conf_id") for x in X):
        raise ValueError(
            "Passed data must be molecules (rdkit.Chem.rdChem.Mol instances) "
            "and each must have conf_id attribute. You can use "
            "ConformerGenerator to add them."
        )
    return X
