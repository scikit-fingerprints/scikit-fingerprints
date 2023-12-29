from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as spsparse

from skfp.fingerprints.base import FingerprintTransformer


class AtomPairFingerprint(FingerprintTransformer):
    def __init__(
        self,
        min_distance: int = 1,
        max_distance: int = 30,
        include_chirality: bool = False,
        use_2D: bool = True,
        count_simulation: bool = True,
        count_bounds: Optional[List] = None,
        fp_size: int = 2048,
        n_jobs: int = None,
        sparse: bool = False,
        count: bool = False,
        verbose: int = 0,
        random_state: int = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            sparse=sparse,
            count=count,
            verbose=verbose,
            random_state=random_state,
        )
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.include_chirality = include_chirality
        self.use_2D = use_2D
        self.count_simulation = count_simulation
        self.count_bounds = count_bounds
        self.fp_size = fp_size

    def _get_generator(self):
        from rdkit.Chem.rdFingerprintGenerator import GetAtomPairGenerator

        return GetAtomPairGenerator(
            minDistance=self.min_distance,
            maxDistance=self.max_distance,
            includeChirality=self.include_chirality,
            use2D=self.use_2D,
            countSimulation=self.count_simulation,
            countBounds=self.count_bounds,
            fpSize=self.fp_size,
        )

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        X = self._validate_input(X)
        return self._generate_fingerprints(X)
