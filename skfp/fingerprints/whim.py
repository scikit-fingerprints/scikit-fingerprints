from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class WHIMFingerprint(FingerprintTransformer):
    def __init__(
        self,
        clip_val: int = np.iinfo(np.int32).max,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.clip_val = clip_val

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, List[str]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import CalcWHIM

        X = self._validate_input(X, require_conf_ids=True)
        X = [CalcWHIM(mol, confId=mol.conf_id) for mol in X]
        X = np.minimum(X, self.clip_val)
        return csr_array(X) if self.sparse else np.array(X)
