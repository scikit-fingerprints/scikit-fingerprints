from typing import Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer
from skfp.validators import ensure_mols, require_mols_with_conf_ids


class PharmacophoreFingerprint(FingerprintTransformer):
    def __init__(
        self,
        variant: str = "raw_bits",
        fp_size: int = 2048,
        use_3D: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        if variant not in ["bit", "count", "raw_bits"]:
            raise ValueError("Variant must be one of: 'bit', 'count', 'raw_bits'")

        super().__init__(
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.variant = variant
        self.fp_size = fp_size
        self.use_3D = use_3D

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem import Get3DDistanceMatrix
        from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
        from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint

        factory = Gobbi_Pharm2D.factory

        if not self.use_3D:
            X = ensure_mols(X)
            X = [Gen2DFingerprint(x, factory) for x in X]
        else:
            X = require_mols_with_conf_ids(X)
            X = [
                Gen2DFingerprint(
                    x, factory, dMat=Get3DDistanceMatrix(x, confId=x.conf_id)
                )
                for x in X
            ]

        if self.variant in ["bit", "count"]:
            # X at this point is a list of RDKit fingerprints, but MyPy doesn't get it
            X = self._hash_fingerprint_bits(
                X,  # type: ignore
                fp_size=self.fp_size,
                count=(self.variant == "count"),
                sparse=self.sparse,
            )

        return csr_array(X) if self.sparse else np.array(X)
