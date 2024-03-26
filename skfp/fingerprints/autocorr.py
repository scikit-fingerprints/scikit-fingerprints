from typing import Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer
from skfp.validators import ensure_mols, require_mols_with_conf_ids


class AutocorrFingerprint(FingerprintTransformer):
    """Autocorrelation fingerprint."""

    def __init__(
        self,
        use_3D: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
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
