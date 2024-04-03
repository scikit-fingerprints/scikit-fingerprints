from typing import Any, Sequence

from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles


def ensure_mols(X: Sequence[Any]) -> Sequence[Mol]:
    if not all(isinstance(x, (Mol, str)) for x in X):
        raise ValueError("Passed values must be either rdkit.Chem.rdChem.Mol or SMILES")

    X = [MolFromSmiles(x) if isinstance(x, str) else x for x in X]
    return X


def ensure_smiles(X: Sequence[Any]) -> Sequence[str]:
    if not all(isinstance(x, (Mol, str)) for x in X):
        raise ValueError("Passed values must be SMILES strings")
    X = [MolToSmiles(x) if isinstance(x, Mol) else x for x in X]
    return X


def require_mols_with_conf_ids(X: Sequence[Any]) -> Sequence[Mol]:
    if not all(isinstance(x, Mol) and hasattr(x, "conf_id") for x in X):
        raise ValueError(
            "Passed data must be molecules (rdkit.Chem.rdChem.Mol instances) "
            "and each must have conf_id attribute. You can use "
            "ConformerGenerator to add them."
        )
    return X
