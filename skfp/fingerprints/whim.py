from collections.abc import Sequence
from numbers import Real
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils import Interval

from skfp.validators import require_mols_with_conf_ids

from .base import FingerprintTransformer


class WHIMFingerprint(FingerprintTransformer):
    """WHIM fingerprint."""

    _parameter_constraints: dict = {
        **FingerprintTransformer._parameter_constraints,
        "clip_val": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        clip_val: int = np.iinfo(np.int32).max,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=114,
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.clip_val = clip_val

    def _calculate_fingerprint(self, X: Sequence[Mol]) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import CalcWHIM

        X = require_mols_with_conf_ids(X)
        X = [CalcWHIM(mol, confId=mol.conf_id) for mol in X]
        X = np.clip(X, -self.clip_val, self.clip_val)
        return csr_array(X) if self.sparse else np.array(X)
