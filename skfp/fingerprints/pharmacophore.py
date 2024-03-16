from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class PharmacophoreFingerprint(FingerprintTransformer):
    def __init__(
        self,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, List[str]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
        from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint

        X = self._validate_input(X)
        factory = Gobbi_Pharm2D.factory
        from rdkit.Chem import Get3DDistanceMatrix

        X = [Gen2DFingerprint(x, factory, dMat=Get3DDistanceMatrix(x)) for x in X]

        if self.sparse:
            return csr_array(X)
        else:
            return np.array(X)
