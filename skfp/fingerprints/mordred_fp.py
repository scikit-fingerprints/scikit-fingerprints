from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from mordred import Calculator, descriptors
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import ensure_mols


class MordredFingerprint(BaseFingerprintTransformer):
    """Mordred fingerprint."""

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "use_3D": ["boolean"],
    }

    def __init__(
        self,
        use_3D: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        n_features_out = 1826 if use_3D else 1613
        super().__init__(
            n_features_out=n_features_out,
            requires_conformers=use_3D,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.use_3D = use_3D

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        X = ensure_mols(X)

        calc = Calculator(descriptors, ignore_3D=not self.use_3D)
        X = [calc(mol) for mol in X]

        return (
            csr_array(X, dtype=np.float32)
            if self.sparse
            else np.array(X, dtype=np.float32)
        )
