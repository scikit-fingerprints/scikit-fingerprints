from typing import Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer
from skfp.validators import require_mols_with_conf_ids


class RDFFingerprint(FingerprintTransformer):
    """RDF fingerprint."""

    def __init__(
        self,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import CalcRDF

        X = require_mols_with_conf_ids(X)
        X = [CalcRDF(mol, confId=mol.conf_id) for mol in X]
        return csr_array(X) if self.sparse else np.array(X)
