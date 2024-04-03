from numbers import Integral
from typing import Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils import Interval

from skfp.validators import ensure_mols

from .base import FingerprintTransformer


class PatternFingerprint(FingerprintTransformer):
    """Pattern fingerprint."""

    _parameter_constraints: dict = {
        **FingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "tautomers": ["boolean"],
    }

    def __init__(
        self,
        fp_size: int = 2048,
        tautomers: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            sparse=sparse,
            n_jobs=n_jobs,
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
        return csr_array(X) if self.sparse else np.array(X)
