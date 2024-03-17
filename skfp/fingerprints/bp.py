from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class BPFingerprint(FingerprintTransformer):
    def __init__(
        self,
        fp_size: int = 2048,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.fp_size = fp_size

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, List[str]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint
        from rdkit.DataStructs import SparseBitVect
        from rdkit.DataStructs.cDataStructs import FoldFingerprint

        X = self._validate_input(X)

        X = [GetBPFingerprint(x) for x in X]
        X = [SparseBitVect(size=self.fp_size).SetBitsFromList(x.ToList()) for x in X]
        X = [FoldFingerprint(x) for x in X]

        return X
        # return csr_array(X) if self.sparse else np.array(X)


if __name__ == "__main__":
    smiles = pd.read_csv("hiv_mol.csv")["smiles"][:10]
    bp = BPFingerprint()
    res = bp.transform(smiles)
    print(type(res[0]))
    print(res[0])
