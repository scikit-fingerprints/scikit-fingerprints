from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as spsparse

from skfp.fingerprints.base import FingerprintTransformer


class AvalonFingerprint(FingerprintTransformer):
    def __init__(
        self,
        n_bits: int = 512,
        is_query: bool = False,
        bit_flags: int = 15761407,
        reset_vect: bool = False,
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
        self.n_bits = n_bits
        self.is_query = is_query
        self.bit_flags = bit_flags
        self.reset_vect = reset_vect

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        X = self._validate_input(X)
        from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP, GetAvalonFP

        if self.count:
            X = [
                GetAvalonCountFP(
                    x,
                    nBits=self.n_bits,
                    isQuery=self.is_query,
                    bitFlags=self.bit_flags,
                ).ToList()  # should be sparse or numpy
                for x in X
            ]
        else:
            X = [
                GetAvalonFP(
                    x,
                    nBits=self.n_bits,
                    isQuery=self.is_query,
                    bitFlags=self.bit_flags,
                    resetVect=self.reset_vect,
                )
                for x in X
            ]

        if self.sparse:
            return spsparse.csr_array(X)
        else:
            return np.array(X)
