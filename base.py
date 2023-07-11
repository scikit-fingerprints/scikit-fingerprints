from abc import ABC, abstractmethod

import joblib
from joblib import delayed, effective_n_jobs, Parallel
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdchem import Mol

legal_result_types = {"bit", "sparse", "count", "sparse_count"}


class FingerprintTransformer(ABC, TransformerMixin, BaseEstimator):
    def __init__(self, result_vector_type: str = "bit", n_jobs: int = None):
        """
        result_vector_tape has to be one of the following:
        bit, spares, count, sparse_count
        """
        assert result_vector_type in legal_result_types
        self.n_jobs = effective_n_jobs(n_jobs)
        self.result_vector_type = result_vector_type
        self.fp_generator_function = None
        self.fp_generator_args = {}

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        :param X: np.array or DataFrame of rdkit.Mol objects
        :return: np.array of calculated fingerprints for each molecule
        """

        if self.n_jobs == 1:
            return self._calculate_fingerprint(X)
        else:
            batch_size = max(len(X) // self.n_jobs, 1)

            args = (
                X[i : i + batch_size] for i in range(0, len(X), batch_size)
            )

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._calculate_fingerprint)(X_sub) for X_sub in args
            )

            return np.concatenate(results)

    @abstractmethod
    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
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
