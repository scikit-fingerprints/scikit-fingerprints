from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as spsparse

from skfp.fingerprints.base import FingerprintTransformer


class AvalonFingerprint(FingerprintTransformer):
    def __init__(
        self,
        nBits=512,
        isQuery: bool = False,
        bitFlags: int = 15761407,
        resetVect: bool = False,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
        random_state: int = 0,
        count: bool = False,
    ):
        super().__init__(
            n_jobs=n_jobs,
            sparse=sparse,
            verbose=verbose,
            random_state=random_state,
            count=count,
        )
        self.nBits = nBits
        self.isQuery = isQuery
        self.bitFlags = bitFlags
        self.resetVect = resetVect

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        X = self._validate_input(X)
        from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP, GetAvalonFP

        if self.count:
            X = [
                GetAvalonCountFP(
                    x,
                    nBits=self.nBits,
                    isQuery=self.isQuery,
                    bitFlags=self.bitFlags,
                ).ToList()  # should be sparse or numpy
                for x in X
            ]
        else:
            X = [
                GetAvalonFP(
                    x,
                    nBits=self.nBits,
                    isQuery=self.isQuery,
                    bitFlags=self.bitFlags,
                    resetVect=self.resetVect,
                )
                for x in X
            ]

        if self.sparse:
            return spsparse.csr_array(X)
        else:
            return np.array(X)
