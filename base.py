from abc import ABC

import joblib
from joblib import Parallel, delayed
from typing import Union

import numpy as np
import pandas as pd

from featurizers.descriptors import fp_descriptors


class FingerprintTransformer(ABC):
    def __init__(self, n_jobs):
        if n_jobs == -1:
            self.n_jobs = joblib.cpu_count()
        else:
            self.n_jobs = n_jobs

        self.fp_args = None
        self.fp_descriptor_name: str = ""

    def _transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Helper function to be executed in each sub-process.

        :param X: subset of original X data
        :return: np.array containing calculated fingerprints for each molecule
        """
        result = np.array(
            [
                fp_descriptors[self.fp_descriptor_name](x, **self.fp_args)
                for x in X
            ]
        )
        return result

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        :param X: np.array or DataFrame of rdkit.Mol objects
        :return: np.array of calculated fingerprints for each molecule
        """
        batch_size = len(X) // self.n_jobs

        if batch_size == 0:
            batch_size = 1

        args = [X[i : i + batch_size] for i in range(0, len(X), batch_size)]

        with joblib.parallel_backend("loky", n_jobs=self.n_jobs):
            results = Parallel()(
                delayed(self._transform)(X_sub) for X_sub in args
            )

        return np.concatenate(results)
