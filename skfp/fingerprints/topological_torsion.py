from typing import List, Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class TopologicalTorsionFingerprint(FingerprintTransformer):
    def __init__(
        self,
        fp_size: int = 2048,
        include_chirality: bool = False,
        torsion_atom_count: int = 4,
        atom_invariants_generator: Optional[List] = None,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.fp_size = fp_size
        self.include_chirality = include_chirality
        self.torsion_atom_count = torsion_atom_count
        self.atom_invariants_generator = atom_invariants_generator

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdFingerprintGenerator import GetTopologicalTorsionGenerator

        X = self._validate_input(X)

        gen = GetTopologicalTorsionGenerator(
            includeChirality=self.include_chirality,
            torsionAtomCount=self.torsion_atom_count,
            countSimulation=self.count,
            atomInvariantsGenerator=self.atom_invariants_generator,
            fpSize=self.fp_size,
        )

        if self.count:
            X = [gen.GetCountFingerprintAsNumPy(x) for x in X]
        else:
            X = [gen.GetFingerprintAsNumPy(x) for x in X]

        return csr_array(X) if self.sparse else np.array(X)
