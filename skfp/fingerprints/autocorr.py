from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import ensure_mols, require_mols_with_conf_ids


class AutocorrFingerprint(BaseFingerprintTransformer):
    """Autocorrelation fingerprint."""

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "use_3D": ["boolean"],
    }

    def __init__(
        self,
        use_3D: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        n_features_out = 80 if use_3D else 192
        super().__init__(
            n_features_out=n_features_out,
            requires_conformers=use_3D,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
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
            X = [CalcAUTOCORR3D(mol, confId=mol.GetIntProp("conf_id")) for mol in X]

        return csr_array(X) if self.sparse else np.array(X)
