from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array, dok_array

from skfp.fingerprints.base import FingerprintTransformer


class BTFingerprint(FingerprintTransformer):
    def __init__(
        self,
        fp_size: int = 2048,
        count: bool = False,
        sparse: bool = False,
        n_jobs: int = None,
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
        from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint

        X = self._validate_input(X)

        X = [GetBPFingerprint(mol) for mol in X]

        # hash raw fingerprint bits to array values
        shape = (len(X), self.fp_size)
        arr = dok_array(shape, dtype=int) if self.sparse else np.zeros(shape, dtype=int)

        for idx, x in enumerate(X):
            for fp_bit, count in x.GetNonzeroElements().items():
                if self.count:
                    arr[idx, fp_bit % self.fp_size] += count
                else:
                    arr[idx, fp_bit % self.fp_size] = 1

        return arr.tocsr() if self.sparse else arr
