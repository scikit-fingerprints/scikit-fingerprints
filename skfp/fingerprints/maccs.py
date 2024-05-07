from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import ensure_mols


class MACCSFingerprint(BaseFingerprintTransformer):
    """MACCS fingerprint."""

    def __init__(
        self,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=167,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint

        X = ensure_mols(X)
        X = [GetMACCSKeysFingerprint(mol) for mol in X]

        if self.sparse:
            return csr_array(X, dtype=np.uint8)
        else:
            return np.array(X, dtype=np.uint8)
