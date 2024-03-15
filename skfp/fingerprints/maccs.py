from typing import Union, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class MACCSFingerprint(FingerprintTransformer):
    def __init__(
        self,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
        random_state: int = 0,
    ):
        super().__init__(
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, List[str]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint

        X = self._validate_input(X)

        X = [GetMACCSKeysFingerprint(x) for x in X]
        if self.sparse:
            return csr_array(X)
        else:
            return np.array(X)
