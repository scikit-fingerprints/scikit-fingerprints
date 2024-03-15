from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class AtomPairFingerprint(FingerprintTransformer):
    def __init__(
        self,
        fp_size: int = 2048,
        min_distance: int = 1,
        max_distance: int = 30,
        include_chirality: bool = False,
        use_2D: bool = True,
        count: bool = False,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
        random_state: int = 0,
    ):
        super().__init__(
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
        self.fp_size = fp_size
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.include_chirality = include_chirality
        self.use_2D = use_2D

    def _get_generator(self):
        from rdkit.Chem.rdFingerprintGenerator import GetAtomPairGenerator

        return GetAtomPairGenerator(
            fpSize=self.fp_size,
            minDistance=self.min_distance,
            maxDistance=self.max_distance,
            includeChirality=self.include_chirality,
            use2D=self.use_2D,
            countSimulation=self.count,
        )

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, List[str]]
    ) -> Union[np.ndarray, csr_array]:
        X = self._validate_input(X)
        return self._generate_fingerprints(X)
