from typing import Optional, Sequence, Union

import numpy as np
from mordred import Calculator, descriptors
from mordred.error import Missing
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer
from skfp.validators import ensure_mols


class MordredFingerprint(FingerprintTransformer):
    """Mordred fingerprint."""

    def __init__(
        self,
        ignore_3D: bool = True,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.ignore_3D = ignore_3D

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        X = ensure_mols(X)

        calc = Calculator(descriptors, ignore_3D=self.ignore_3D)
        X = [calc(x) for x in X]

        return (
            csr_array(X, dtype=np.float32)
            if self.sparse
            else np.array(X, dtype=np.float32)
        )
