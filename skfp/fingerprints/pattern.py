from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils import Interval

from skfp.bases.base_fp_transformer import BaseFingerprintTransformer
from skfp.validators import ensure_mols


class PatternFingerprint(BaseFingerprintTransformer):
    """Pattern fingerprint."""

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "tautomers": ["boolean"],
    }

    def __init__(
        self,
        fp_size: int = 2048,
        tautomers: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.fp_size = fp_size
        self.tautomers = tautomers

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdmolops import PatternFingerprint as RDKitPatternFingerprint

        X = ensure_mols(X)
        X = [
            RDKitPatternFingerprint(
                x, fpSize=self.fp_size, tautomerFingerprints=self.tautomers
            )
            for x in X
        ]

        if self.sparse:
            return csr_array(X, dtype=np.uint8)
        else:
            return np.array(X, dtype=np.uint8)
