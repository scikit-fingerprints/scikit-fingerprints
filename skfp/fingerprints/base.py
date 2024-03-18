from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd
import scipy.sparse
from joblib import Parallel, delayed, effective_n_jobs
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdchem import Mol
from scipy.sparse import csr_array
from sklearn.base import BaseEstimator, TransformerMixin

from skfp.utils import ProgressParallel

"""
If during multiprocessing occurs MaybeEncodingError, first check if there isn't thrown any exception inside
worker function! (That error isn't very informative and this tip might save you a lot of time)
"""

"""
fp_descriptors need to be inside _calculate_fingerprint() of a specific class (cannot be defined inside the __init__() of
that class), otherwise pickle gets angry:
TypeError: cannot pickle 'Boost.Python.function' object
"""


class FingerprintTransformer(ABC, TransformerMixin, BaseEstimator):
    def __init__(
        self,
        count: bool = False,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
        random_state: int = 0,
    ):
        self.count = count
        self.sparse = sparse
        self.n_jobs = effective_n_jobs(n_jobs)
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X: Union[pd.DataFrame, np.ndarray, List[str]]):
        """
        :param X: np.array or DataFrame of rdkit.Mol objects
        :return: np.array or sparse array of calculated fingerprints for each molecule
        """

        if self.n_jobs == 1:
            return self._calculate_fingerprint(X)
        else:
            batch_size = max(len(X) // self.n_jobs, 1)

            args = (X[i : i + batch_size] for i in range(0, len(X), batch_size))

            if self.verbose > 0:
                total = min(self.n_jobs, len(X))
                parallel = ProgressParallel(n_jobs=self.n_jobs, total=total)
            else:
                parallel = Parallel(n_jobs=self.n_jobs)

            results = parallel(
                delayed(self._calculate_fingerprint)(X_sub) for X_sub in args
            )

            return scipy.sparse.vstack(results) if self.sparse else np.vstack(results)

    @abstractmethod
    def _calculate_fingerprint(
        self, X: Union[np.ndarray]
    ) -> Union[np.ndarray, csr_array]:
        """
        Calculate fingerprints for a given input batch.

        :param X: subset of original X data
        :return: array containing calculated fingerprints for each molecule
        """
        pass

    def _validate_input(self, X: List, smiles_only: bool = False) -> List[Mol]:
        if smiles_only:
            if not all(isinstance(x, str) for x in X):
                raise ValueError("Passed values must be SMILES strings")
            return X

        if not all(isinstance(x, Mol) or isinstance(x, str) for x in X):
            raise ValueError(
                "Passed value must be either rdkit.Chem.rdChem.Mol or SMILES"
            )

        X = [MolFromSmiles(x) if isinstance(x, str) else x for x in X]
        return X
