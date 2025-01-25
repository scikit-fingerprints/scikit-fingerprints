import functools
from collections.abc import Sequence
from typing import Any, Callable

from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles
from rdkit.Chem.PropertyMol import PropertyMol


def ensure_mols(X: Sequence[Any]) -> list[Mol]:
    """
    Ensure that all input sequence elements are RDKit ``Mol`` objects. Requires
    all input elements to be of the same type: string (SMILES strings) or ``Mol``.
    In case of SMILES strings, they are converted to RDKit ``Mol`` objects with
    default settings.
    """
    if not all(isinstance(x, (Mol, PropertyMol, str)) for x in X):
        types = {type(x) for x in X}
        raise ValueError(
            f"Passed values must be RDKit Mol objects or SMILES strings,"
            f"got types: {types}"
        )

    mols = [MolFromSmiles(x) if isinstance(x, str) else x for x in X]

    if any(x is None for x in mols):
        idx = mols.index(None)
        raise ValueError(f"Could not parse '{X[idx]}' at index {idx} as molecule")

    return mols


def ensure_smiles(X: Sequence[Any]) -> list[str]:
    """
    Ensure that all input sequence elements are SMILES strings. Requires all input
    elements to be of the same type: string (SMILES strings) or ``Mol``. In case of
    RDKit ``Mol`` objects, they are converted to SMILES strings with default settings.
    """
    if not all(isinstance(x, (Mol, PropertyMol, str)) for x in X):
        types = {type(x) for x in X}
        raise ValueError(f"Passed values must be SMILES strings, got types: {types}")

    X = [MolToSmiles(x) if isinstance(x, Mol) else x for x in X]
    return X


def require_mols(X: Sequence[Any]) -> None:
    """
    Check that all inputs are RDKit ``Mol`` objects, raises ValueError otherwise.
    """
    for idx, x in enumerate(X):
        if not isinstance(x, (Mol, PropertyMol)):
            raise TypeError(
                f"Passed values must be RDKit ``Mol`` objects, got type {type(x)} at index {idx}"
            )


def require_mols_with_conf_ids(X: Sequence[Any]) -> Sequence[Mol]:
    """
    Check that all inputs are RDKit ``Mol`` objects with ``"conf_id"`` property
    set, i.e. with conformers computed and properly identified. Raises ValueError
    otherwise.
    """
    if not all(isinstance(x, (Mol, PropertyMol)) and x.HasProp("conf_id") for x in X):
        raise TypeError(
            "Passed data must be molecules (RDKit Mol objects) "
            "and each must have conf_id property set. "
            "You can use ConformerGenerator to add them."
        )
    return X


def require_strings(X: Sequence[Any]) -> None:
    """
    Check that all inputs are strings, raises ValueError otherwise.
    """
    for idx, x in enumerate(X):
        if not isinstance(x, str):
            raise TypeError(
                f"Passed values must be strings, got type {type(x)} at index {idx}"
            )


def require_atoms(min_atoms: int = 1) -> Callable:
    """
    Decorator for functions operating on single molecule. Ensures it is
    nonempty (by default) or has at least the specified number of atoms, raises ValueError otherwise.
    """  # noqa: D401

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(mol: Mol, *args, **kwargs):
            if mol.GetNumAtoms() < min_atoms:
                raise ValueError(
                    f"The molecule must have at least {min_atoms} atom(s), {func.__name__} cannot be calculated."
                )
            return func(mol, *args, **kwargs)

        return wrapper

    return decorator
