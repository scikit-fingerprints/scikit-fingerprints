from typing import Union, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class EStateFingerprint(FingerprintTransformer):
    def __init__(
        self,
        variant: str = "sum",
        sparse: bool = False,
        count: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        if variant not in ["sum", "binary"]:
            raise ValueError("Variant must be 'sum' or 'binary'")

        if self.count and variant == "sum":
            raise ValueError("Count version only available with 'sum' variant")

        super().__init__(
            n_jobs=n_jobs,
            sparse=sparse,
            verbose=verbose,
            count=count,
        )
        self.variant = variant

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, List[str]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.EState.Fingerprinter import FingerprintMol

        X = self._validate_input(X)

        X = np.array([FingerprintMol(x) for x in X])
        if self.count:
            X = X[:, 0]
        elif self.variant == "binary":
            X = X[:, 0] > 0
        else:
            X = X[:, 1]

        if self.sparse:
            return csr_array(X)
        else:
            return X
