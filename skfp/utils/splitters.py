import random
from collections import defaultdict
from typing import Optional
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def scaffold_split(smiles_list: list[str],
                   frac_train: float = 0.8,
                   frac_valid: float = 0.1,
                   frac_test: float = 0.1,
                   seed: Optional[int] = None) -> tuple[list[str], list[str], list[str]]:
    """
    Generates scaffolds for a list of SMILES strings.

    Args:
    smiles_list (List[str]): A list of SMILES strings.
    frac_train (float): Fraction of the dataset to be used as the training set.
    frac_valid (float): Fraction of the dataset to be used as the validation set.
    frac_test (float): Fraction of the dataset to be used as the test set.
    seed (Optional[int]): Random seed for reproducibility.

    Returns:
    Tuple[List[str], List[str], List[str]]: A tuple containing the training, validation, and test sets.
    """
    if seed is not None:
        random.seed(seed)

    scaffolds: dict[str, list[str]] = {}
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        if scaffold_smiles not in scaffolds:
            scaffolds[scaffold_smiles] = []
        scaffolds[scaffold_smiles].append(smile)

    scaffold_sets = list(scaffolds.values())
    random.shuffle(scaffold_sets)

    n_total: int = len(smiles_list)
    n_train: int = int(frac_train * n_total)
    n_valid : int = len(frac_valid * n_total)
    n_test = n_total - n_train - n_valid

    train_set: list[str] = []
    valid_set: list[str] = []
    test_set: list[str] = []

    for scaffold_set in scaffold_sets:
        if len(train_set) + len(scaffold_set) <= n_train:
            train_set.extend(scaffold_set)
        elif len(valid_set) + len(scaffold_set) <= n_valid:
            valid_set.extend(scaffold_set)
        else:
            test_set.extend(scaffold_set)

    return train_set, valid_set, test_set
