from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse as spsparse

from skfp.fingerprints.base import FingerprintTransformer


class EStateFingerprint(FingerprintTransformer):
    def __init__(
        self,
        variant: str = "sum",
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
        random_state: int = 0,
        count: bool = False,
    ):
        """'variant' argument determines the output type. It can be one 'sum', or 'binary'"""
        super().__init__(
            n_jobs=n_jobs,
            sparse=sparse,
            verbose=verbose,
            random_state=random_state,
            count=count,
        )
        if variant not in ["sum", "binary"]:
            raise ValueError("variant must be 'sum' or 'binary'")
        if self.count and variant == "sum":
            raise ValueError(
                "count fingerprint only available with 'binary' variant"
            )
        self.variant = variant

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        X = self._validate_input(X)
        from rdkit.Chem.EState.Fingerprinter import FingerprintMol

        X = np.array([FingerprintMol(x) for x in X])
        if self.count:
            X = X[:, 0]
        elif self.variant == "binary":
            X = X[:, 0] > 0
        else:
            X = X[:, 1]

        if self.sparse:
            return spsparse.csr_array(X)
        else:
            return X
