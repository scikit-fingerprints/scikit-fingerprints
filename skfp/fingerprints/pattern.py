from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class PatternFingerprint(FingerprintTransformer):
    def __init__(
        self,
        fp_size: int = 2048,
        tautomers: bool = False,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.fp_size = fp_size
        self.tautomers = tautomers

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, List[str]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdmolops import PatternFingerprint as RDKitPatternFingerprint

        X = self._validate_input(X)
        X = [
            RDKitPatternFingerprint(
                x, fpSize=self.fp_size, tautomerFingerprints=self.tautomers
            )
            for x in X
        ]
        return csr_array(X) if self.sparse else np.array(X)
