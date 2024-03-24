from typing import Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer
from skfp.validators import ensure_mols


class EStateFingerprint(FingerprintTransformer):
    """EState fingerprint."""

    def __init__(
        self,
        variant: str = "sum",
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        if variant not in ["bit", "count", "sum"]:
            raise ValueError("Variant must be one of: 'bit', 'count', 'sum'")

        super().__init__(
            n_jobs=n_jobs,
            sparse=sparse,
            verbose=verbose,
        )
        self.variant = variant

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.EState.Fingerprinter import FingerprintMol

        X = ensure_mols(X)

        X = np.array([FingerprintMol(x) for x in X])
        if self.variant == "bit":
            X = (X[:, 0] > 0).astype(int)
        elif self.variant == "count":
            X = (X[:, 0]).astype(int)
        else:  # "sum" variant
            X = X[:, 1]

        return csr_array(X) if self.sparse else np.array(X)
