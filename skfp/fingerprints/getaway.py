from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class GETAWAYFingerprint(FingerprintTransformer):
    def __init__(
        self,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, List[str]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import CalcGETAWAY

        X = self._validate_input(X, require_conf_ids=True, mol_from_smiles=False)
        X = [CalcGETAWAY(mol, confId=mol.conf_id) for mol in X]
        return csr_array(X) if self.sparse else np.array(X)
