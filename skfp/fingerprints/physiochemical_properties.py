from typing import Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer
from skfp.validators import ensure_mols


class PhysiochemicalPropertiesFingerprint(FingerprintTransformer):
    """Physiochemical properties fingerprint."""

    def __init__(
        self,
        fp_size: int = 2048,
        variant: str = "BP",
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        if variant not in ["BP", "BT"]:
            raise ValueError("Variant must be one of: 'BP', 'BT'")

        super().__init__(
            n_features_out=fp_size,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
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
