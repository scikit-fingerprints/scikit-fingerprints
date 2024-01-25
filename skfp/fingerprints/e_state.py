from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse as spsparse

from skfp.fingerprints.base import FingerprintTransformer


class EStateFingerprint(FingerprintTransformer):
    """this fingerprint returns 2 arrays:
    The first (of ints) contains the number of times each possible atom type is hit
     The second (of floats) contains the sum of the EState indices for atoms of
    """

    def __init__(
        self,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
        random_state: int = 0,
        count: bool = False,
    ):
        super().__init__(
            n_jobs=n_jobs,
            sparse=sparse,
            verbose=verbose,
            random_state=random_state,
            count=count,
        )

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        X = self._validate_input(X)
        from rdkit.Chem.EState.Fingerprinter import FingerprintMol

        X = np.array([FingerprintMol(x) for x in X])
        X = np.concatenate([X[:, 0], X[:, 1]], axis=1)

        if self.sparse:
            return spsparse.csr_array(X)
        else:
            return X

    def transform(self, X: Union[pd.DataFrame, np.ndarray, list[str]]):
        X = super().transform(X)
        return X[:, :79], X[:, 79:]
