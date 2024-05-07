from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import ensure_mols


class PhysiochemicalPropertiesFingerprint(BaseFingerprintTransformer):
    """Physiochemical properties fingerprint."""

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "variant": [StrOptions({"BP", "BT"})],
    }

    def __init__(
        self,
        fp_size: int = 2048,
        variant: str = "BP",
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.fp_size = fp_size
        self.variant = variant

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint

        X = ensure_mols(X)

        if self.variant == "BP":
            X = [GetBPFingerprint(mol) for mol in X]
        else:  # "BT" variant
            X = [GetBTFingerprint(mol) for mol in X]

        X = self._hash_fingerprint_bits(
            X, fp_size=self.fp_size, count=self.count, sparse=self.sparse
        )

        return X
