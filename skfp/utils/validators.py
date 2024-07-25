import numbers
import numpy as np

from collections.abc import Sequence
from typing import Any, Callable, Union, Sequence

from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles
from rdkit.Chem.PropertyMol import PropertyMol

class Interval:
    def __init__(self, type_, low: float, high: float, closed: str = "both"):
        self.type_ = type_
        self.low = low
        self.high = high
        self.closed = closed

    def __call__(self, value):
        if not isinstance(value, self.type_):
            return False
        if self.closed in ("left", "both") and value < self.low:
            return False
        if self.closed in ("right", "both") and value > self.high:
            return False
        return True

class RealNotInt(numbers.Real):
    def __call__(self, value):
        return isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral)
    
def validate_params(rules: dict[str, list[Any]]):
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            func_params = func.__code__.co_varnames[:func.__code__.co_argcount]
            all_args = dict(zip(func_params, args))
            all_args.update(kwargs)

            for param, value in all_args.items():
                if param in rules:
                    valid = False
                    for rule in rules[param]:
                        if isinstance(rule, Interval):
                            if rule(value):
                                valid = True
                                break
                        elif rule == "boolean" and isinstance(value, bool):
                            valid = True
                            break
                        elif rule == "sequence" and isinstance(value, Sequence):
                            valid = True
                            break
                        elif rule == "list" and isinstance(value, list):
                            valid = True
                            break
                        elif rule == "random_state" and (isinstance(value, numbers.Integral) or value is None):
                            valid = True
                            break
                        elif rule == "array-like" and isinstance(value, (list, tuple, np.ndarray)):
                            valid = True
                            break
                        elif value is None and rule is None:
                            valid = True
                            break
                    if not valid:
                        raise ValueError(f"Invalid value for parameter '{param}': {value}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def ensure_mols(X: Sequence[Any]) -> list[Mol]:
    if not all(isinstance(x, (Mol, PropertyMol, str)) for x in X):
        raise ValueError("Passed values must be either rdkit.Chem.rdChem.Mol or SMILES")

    X = [MolFromSmiles(x) if isinstance(x, str) else x for x in X]
    return X


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
