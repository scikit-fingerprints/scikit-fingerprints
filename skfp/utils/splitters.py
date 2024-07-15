from collections.abc import Sequence
import numpy as np
from typing import Optional, Any

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from validators import ensure_mols

def scaffold_split(molecules: Sequence[Any],
                   frac_train: float = 0.8,
                   frac_valid: float = 0.1,
                   frac_test: float = 0.1,
                   seed: Optional[int] = None) -> tuple[list[int], list[int],  list[int]]:
    
    """
    Splits a list of SMILES or MOL objects into train/val/test using scaffold

    Parameters
    ----------
    molecules: str 
        Sequence representing either SMILES of Mol
    frac_train: float, optional (default 0.8)
        The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
        The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
        The fraction of data to be used for the test split.
    seed: int, optional (default None)
        Random seed to use.

    Returns
    ----------
    tuple[list[int], list[int], list[int]]
        A tuple of train, validation and test indices, each is 
        a list of ints.
    """
    molecules = ensure_mols(molecules)

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if seed is not None:
        np.random.seed(seed=seed)

    scaffolds: dict[str, list[int]] = {}
    
    for ind, molecule in enumerate(molecules):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule,
            includeChirality=True
        )
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    scaffold_sets = [scaffold_set for (_, scaffold_set) 
                    in sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)]


    train_cutoff: int = frac_train * len(molecules)
    valid_cutoff: int = train_cutoff + frac_valid * len(molecules)
    train_ids: list[int] = []
    valid_ids: list[int] = []
    test_ids: list[int] = []

    for scaffold_set in scaffold_sets:
        if len(train_ids) + len(scaffold_set) > train_cutoff:
            if len(train_ids) + len(valid_ids) + len(scaffold_set) > valid_cutoff:
                test_ids += scaffold_set
            else:
                valid_ids += scaffold_set
        else:
            train_ids += scaffold_set

    return train_ids, valid_ids, test_ids
