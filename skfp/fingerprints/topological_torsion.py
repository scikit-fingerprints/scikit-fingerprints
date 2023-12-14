from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as spsparse

from skfp.fingerprints.base import FingerprintTransformer


class TopologicalTorsionFingerprint(FingerprintTransformer):
    def __init__(
        self,
        include_chirality: bool = False,
        torsion_atom_count: int = 4,
        count_simulation: bool = True,
        count_bounds: Optional[List] = None,
        fp_size: int = 2048,
        atom_invariants_generator: Optional[List] = None,
        n_jobs: int = None,
        sparse: bool = False,
        count: bool = False,
        verbose: int = 0,
        random_state: int = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            sparse=sparse,
            count=count,
            verbose=verbose,
            random_state=random_state,
        )
        self.include_chirality = include_chirality
        self.torsion_atom_count = torsion_atom_count
        self.count_simulation = count_simulation
        self.count_bounds = count_bounds
        self.fp_size = fp_size
        self.atom_invariants_generator = atom_invariants_generator

    def _get_generator(self):
        from rdkit.Chem.rdFingerprintGenerator import GetTopologicalTorsionGenerator

        return GetTopologicalTorsionGenerator(
            includeChirality=self.include_chirality,
            torsionAtomCount=self.torsion_atom_count,
            countSimulation=self.count_simulation,
            countBounds=self.count_bounds,
            atomInvariantsGenerator=self.atom_invariants_generator,
            fpSize=self.fp_size,
        )

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        X = self._validate_input(X)
        return self._generate_fingerprints(X)
