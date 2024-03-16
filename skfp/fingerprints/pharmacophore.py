from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class PharmacophoreFingerprint(FingerprintTransformer):
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
        from rdkit.Chem import Get3DDistanceMatrix
        from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
        from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint

        factory = Gobbi_Pharm2D.factory

        if not self.use_3D:
            X = self._validate_input(X)
            X = [Gen2DFingerprint(x, factory) for x in X]
        else:
            X = self._validate_input(X, require_conf_ids=True)
            X = [Gen2DFingerprint(x, factory, dMat=Get3DDistanceMatrix(x)) for x in X]

        return csr_array(X) if self.sparse else np.array(X)
