from typing import Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.validators import ensure_mols, require_mols_with_conf_ids

from .base import FingerprintTransformer


class AutocorrFingerprint(FingerprintTransformer):
    """Autocorrelation fingerprint."""

    _parameter_constraints: dict = {
        **FingerprintTransformer._parameter_constraints,
        "use_3D": ["boolean"],
    }

    def __init__(
        self,
        use_3D: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        n_features_out = 80 if use_3D else 192
        super().__init__(
            n_features_out=n_features_out,
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.use_3D = use_3D

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import CalcAUTOCORR2D, CalcAUTOCORR3D

        if not self.use_3D:
            X = ensure_mols(X)
            X = [CalcAUTOCORR2D(mol) for mol in X]
        else:
            X = require_mols_with_conf_ids(X)
            X = [CalcAUTOCORR3D(mol, confId=mol.conf_id) for mol in X]

        return csr_array(X) if self.sparse else np.array(X)
