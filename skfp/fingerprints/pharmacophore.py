from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import ensure_mols, require_mols_with_conf_ids


class PharmacophoreFingerprint(BaseFingerprintTransformer):
    """Pharmacophore fingerprint."""

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "variant": [StrOptions({"raw_bits", "bit", "count"})],
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "use_3D": ["boolean"],
    }

    def __init__(
        self,
        variant: str = "raw_bits",
        fp_size: int = 2048,
        use_3D: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        n_features_out = 39972 if variant == "raw_bits" else fp_size
        super().__init__(
            n_features_out=n_features_out,
            requires_conformers=use_3D,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
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
            X = [Gen2DFingerprint(mol, factory) for mol in X]
        else:
            X = require_mols_with_conf_ids(X)
            X = [
                Gen2DFingerprint(
                    mol,
                    factory,
                    dMat=Get3DDistanceMatrix(mol, confId=mol.GetIntProp("conf_id")),
                )
                for mol in X
            ]

        if self.variant in {"bit", "count"}:
            # X at this point is a list of RDKit fingerprints, but MyPy doesn't get it
            return self._hash_fingerprint_bits(
                X,  # type: ignore
                fp_size=self.fp_size,
                count=(self.variant == "count"),
                sparse=self.sparse,
            )
        else:
            return (
                csr_array(X, dtype=np.uint8)
                if self.sparse
                else np.array(X, dtype=np.uint8)
            )
