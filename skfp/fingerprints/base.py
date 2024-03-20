from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd
import scipy.sparse
from joblib import Parallel, delayed, effective_n_jobs
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdchem import Mol
from scipy.sparse import csr_array, dok_array
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

    def _validate_input(
        self, X: List, smiles_only: bool = False, require_conf_ids: bool = False
    ) -> List[Mol]:
        if smiles_only:
            if not all(isinstance(x, str) for x in X):
                raise ValueError("Passed values must be SMILES strings")
            return X

        if require_conf_ids:
            if not all(isinstance(x, Mol) and hasattr(x, "conf_id") for x in X):
                raise ValueError(
                    "Passed data must be molecules (rdkit.Chem.rdChem.Mol instances) "
                    "and each must have conf_id attribute. You can use "
                    "ConformerGenerator to add them."
                )
            return X

        if not all(isinstance(x, Mol) or isinstance(x, str) for x in X):
            raise ValueError(
                "Passed value must be either rdkit.Chem.rdChem.Mol or SMILES"
            )

        X = [MolFromSmiles(x) if isinstance(x, str) else x for x in X]
        return X

    def _hash_fingerprint_bits(self, X: List) -> Union[np.ndarray, csr_array]:
        if not hasattr(self, "fp_size"):
            raise AttributeError(
                "Fingerprint hashing requires inheriting classes to have fp_size attribute"
            )

        shape = (len(X), self.fp_size)
        arr = dok_array(shape, dtype=int) if self.sparse else np.zeros(shape, dtype=int)

        for idx, x in enumerate(X):
            for fp_bit, count in x.GetNonzeroElements().items():
                if self.count:
                    arr[idx, fp_bit % self.fp_size] += count
                else:
                    arr[idx, fp_bit % self.fp_size] = 1

        return arr.tocsr() if self.sparse else arr
