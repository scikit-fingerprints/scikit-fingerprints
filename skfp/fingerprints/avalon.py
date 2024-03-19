from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class AvalonFingerprint(FingerprintTransformer):
    def __init__(
        self,
        fp_size: int = 512,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.fp_size = fp_size

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, List[str]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP, GetAvalonFP

        X = self._validate_input(X)

        if self.count:
            X = [GetAvalonCountFP(x, nBits=self.fp_size).ToList() for x in X]
        else:
            X = [GetAvalonFP(x, nBits=self.fp_size) for x in X]

        return csr_array(X) if self.sparse else np.array(X)
