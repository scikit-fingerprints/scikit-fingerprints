from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class BCUTFingerprint(FingerprintTransformer):
    def __init__(
        self,
        use_3D: bool = False,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.use_3D = use_3D

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, List[str]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import CalcAUTOCORR2D, CalcAUTOCORR3D

        if not self.use_3D:
            X = self._validate_input(X)
            X = [CalcAUTOCORR2D(mol) for mol in X]
        else:
            X = self._validate_input(X, require_conf_ids=True)
            X = [CalcAUTOCORR3D(mol, confId=mol.conf_id) for mol in X]

        return csr_array(X) if self.sparse else np.array(X)
