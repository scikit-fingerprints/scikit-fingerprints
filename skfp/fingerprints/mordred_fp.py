from typing import Optional, Sequence, Union

import numpy as np
from mordred import Calculator, descriptors
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer
from skfp.validators import ensure_mols


class MordredFingerprint(FingerprintTransformer):
    """Mordred fingerprint."""

    def __init__(
        self,
        use_3D: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        n_features_out = 1826 if use_3D else 1613
        super().__init__(
            n_features_out=n_features_out,
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.use_3D = use_3D

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        X = ensure_mols(X)

        calc = Calculator(descriptors, ignore_3D=not self.use_3D)
        X = [calc(x) for x in X]

        return (
            csr_array(X, dtype=np.float32)
            if self.sparse
            else np.array(X, dtype=np.float32)
        )
