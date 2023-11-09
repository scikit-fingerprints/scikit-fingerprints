from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as spsparse
from joblib import Parallel, delayed, effective_n_jobs
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdchem import Mol
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from utils.logger import tqdm_joblib


class FingerprintTransformer(ABC, TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_jobs: Optional[int] = None,
        sparse: bool = False,
        count: bool = False,
        verbose: int = 0,
    ):
        self.n_jobs = effective_n_jobs(n_jobs)
        self.sparse = sparse
        self.count = count
        self.verbose = verbose

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X: Union[pd.DataFrame, np.ndarray, list[str]]):
        """
        :param X: np.array or DataFrame of rdkit.Mol objects
        :return: np.array or sparse array of calculated fingerprints for each molecule
        """

        if self.n_jobs == 1:
            return self._calculate_fingerprint(X)
        else:
            batch_size = max(len(X) // self.n_jobs, 1)

            args = (
                X[i : i + batch_size] for i in range(0, len(X), batch_size)
            )

            if self.verbose > 0:
                total_batches = min(self.n_jobs, len(X))

                with tqdm_joblib(
                    tqdm(
                        desc="Calculating fingerprints...", total=total_batches
                    )
                ) as progress_bar:
                    results = Parallel(n_jobs=self.n_jobs)(
                        delayed(self._calculate_fingerprint)(X_sub)
                        for X_sub in args
                    )
            else:
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._calculate_fingerprint)(X_sub)
                    for X_sub in args
                )

            if isinstance(results[0], spsparse.csr_array):
                return spsparse.vstack(results)
            else:
                return np.concatenate(results)

    @abstractmethod
    def _calculate_fingerprint(
        self, X: Union[np.ndarray]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        """
        Helper function to be executed in each sub-process.

        :param X: subset of original X data
        :return: np.array containing calculated fingerprints for each molecule
        """
        pass

    def _validate_input(self, X: List):
        if not all(
            [
                isinstance(molecule, Mol) or type(molecule) == str
                for molecule in X
            ]
        ):
            raise ValueError(
                "Passed value is neither rdkit.Chem.rdChem.Mol nor SMILES"
            )

        X = [MolFromSmiles(x) if type(x) == str else x for x in X]
        return X

    def _get_generator(self):
        """
        Function that creates a generator object in each sub-process.

        :return: rdkit fingerprint generator for current fp_generator_kwargs parameter
        """
        pass

    def _generate_fingerprints(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        fp_generator = self._get_generator()

        if self.count:
            X = [fp_generator.GetCountFingerprintAsNumPy(x) for x in X]
        else:
            X = [fp_generator.GetFingerprintAsNumPy(x) for x in X]

        if self.sparse:
            return spsparse.csr_array(X)
        else:
            return np.array(X)
