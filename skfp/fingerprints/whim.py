from collections.abc import Sequence
from numbers import Real
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import require_mols_with_conf_ids


class WHIMFingerprint(BaseFingerprintTransformer):
    """WHIM fingerprint."""

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "clip_val": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        clip_val: int = np.iinfo(np.int32).max,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=114,
            requires_conformers=True,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.clip_val = clip_val

    def _calculate_fingerprint(self, X: Sequence[Mol]) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import CalcWHIM

        X = require_mols_with_conf_ids(X)
        X = [CalcWHIM(mol, confId=mol.GetIntProp("conf_id")) for mol in X]
        X = np.clip(X, -self.clip_val, self.clip_val)
        return csr_array(X) if self.sparse else np.array(X)
