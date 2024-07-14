import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Optional

def scaffold_split(smiles_list: list[str],
                   frac_train: float = 0.8,
                   frac_valid: float = 0.1,
                   frac_test: float = 0.1,
                   seed: Optional[int] = None) -> tuple[
                       list[int], 
                       list[int],   
                       list[int]]:
    """
    Splits a list of SMILES into train/val/test using scaffold

    TODO: 
    - Allow the method to accept types other than SMILES string;
    - Add parameter to configure frqeuency of logging

    Parameters
    ----------
    smiles_list: str 
        List of SMILES strings to be split.
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
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if seed is not None:
        np.random.seed(seed=seed)

    scaffolds: dict[str, list[int]] = {}
    data_len: int = len(smiles_list)
    
    for ind, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=True
            )
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)

    scaffold_sets: list[list[int]] = [scaffold_set for (scaffold, scaffold_set) 
                                    in sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)]

    train_cutoff: int = frac_train * len(smiles_list)
    valid_cutoff: int = train_cutoff + frac_valid * len(smiles_list)
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
