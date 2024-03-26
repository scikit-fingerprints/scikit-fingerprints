from typing import Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer
from skfp.validators import require_mols_with_conf_ids


class WHIMFingerprint(FingerprintTransformer):
    """WHIM fingerprint."""

    def __init__(
        self,
        clip_val: int = np.iinfo(np.int32).max,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.clip_val = clip_val

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import CalcWHIM

        X = require_mols_with_conf_ids(X)
        X = [CalcWHIM(mol, confId=mol.conf_id) for mol in X]
        X = np.clip(X, -self.clip_val, self.clip_val)
        return csr_array(X) if self.sparse else np.array(X)
