from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.validators import require_mols_with_conf_ids

from .base import FingerprintTransformer


class MORSEFingerprint(FingerprintTransformer):
    """MORSE fingerprint."""

    def __init__(
        self,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=224,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import CalcMORSE

        X = require_mols_with_conf_ids(X)
        X = [CalcMORSE(mol, confId=mol.GetIntProp("conf_id")) for mol in X]
        return csr_array(X) if self.sparse else np.array(X)
