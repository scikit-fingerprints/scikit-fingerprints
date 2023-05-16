from abc import ABC, abstractmethod
import multiprocessing as mp
from typing import Union

import numpy as np
import pandas as pd


class FingerprintTransformer(ABC):
    def __init__(self, n_jobs):
        if n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs

        self.fp_descriptor = None
        self.fp_args = None

    @abstractmethod
    def _transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        pass

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        :param X: np.array or DataFrame of rdkit.Mol objects
        :return: np.array of calculated fingerprints for each molecule
        """
        batch_size = len(X) // self.n_jobs

        if batch_size == 0:
            batch_size = 1

        with mp.Pool(processes=self.n_jobs) as pool:
            args = [
                X[i : i + batch_size] for i in range(0, len(X), batch_size)
            ]
            results = pool.map(self._transform, args)

        return np.concatenate(results)
