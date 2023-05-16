import multiprocessing as mp
from typing import Union

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from base import FingerprintTransformer


class MorganFingerprintAsBitVect(FingerprintTransformer):
    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        use_chirality: bool = False,
        use_bond_types: bool = True,
        use_features: bool = False,
        n_jobs: int = 1,
    ):
        self.radius = radius
        self.n_bits = n_bits
        self.use_chirality = use_chirality
        self.use_bond_types = use_bond_types
        self.use_features = use_features

        if n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs

    def _transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        If during multiprocessing occurs MaybeEncodingError, first check if there isn't thrown any exception inside
        worker function! (That error isn't very informative and this tip might save you a lot of time)
        """
        result = np.array(
            [
                GetMorganFingerprintAsBitVect(
                    x,
                    self.radius,
                    nBits=self.n_bits,
                    useChirality=self.use_chirality,
                    useBondTypes=self.use_bond_types,
                    useFeatures=self.use_features,
                )
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

        with mp.Pool(processes=self.n_jobs) as pool:
            args = [
                X[i : i + batch_size] for i in range(0, len(X), batch_size)
            ]
            results = pool.map(self._transform, args)

        return np.concatenate(results)
