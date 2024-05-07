from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import ensure_mols


class EStateFingerprint(BaseFingerprintTransformer):
    """EState fingerprint."""

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "variant": [StrOptions({"bit", "count", "sum"})],
    }

    def __init__(
        self,
        variant: str = "sum",
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=79,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.variant = variant

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.EState.Fingerprinter import FingerprintMol

        X = ensure_mols(X)

        X = np.array([FingerprintMol(mol) for mol in X])
        if self.variant == "bit":
            X = (X[:, 0] > 0).astype(np.uint8)
        elif self.variant == "count":
            X = (X[:, 0]).astype(np.uint32)
        else:  # "sum" variant
            X = X[:, 1]

        return csr_array(X) if self.sparse else np.array(X)
