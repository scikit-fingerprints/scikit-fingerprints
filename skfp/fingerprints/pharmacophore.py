from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse as spsparse

from skfp.fingerprints.base import FingerprintTransformer


class PharmacophoreFingerprint(FingerprintTransformer):
    def __init__(
        self,
        three_dimensional: bool = False,
        random_state: int = 0,
        sparse: bool = False,
        count: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            sparse=sparse,
            count=count,
            verbose=verbose,
            random_state=random_state,
        )
        self.three_dimensional = three_dimensional

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        X = self._validate_input(X)

        from rdkit.Chem import Get3DDistanceMatrix
        from rdkit.Chem.AllChem import EmbedMolecule
        from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
        from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint

        factory = Gobbi_Pharm2D.factory

        if self.three_dimensional:
            for x in X:
                EmbedMolecule(x)
            X = [
                Gen2DFingerprint(x, factory, dMat=Get3DDistanceMatrix(x))
                for x in X
            ]
        else:
            X = [Gen2DFingerprint(x, factory) for x in X]

        if self.sparse:
            return spsparse.csr_array(X)
        else:
            return np.array(X)
