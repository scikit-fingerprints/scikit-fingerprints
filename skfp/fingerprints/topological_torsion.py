from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import ensure_mols


class TopologicalTorsionFingerprint(BaseFingerprintTransformer):
    """Topological torsion fingerprint."""

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "include_chirality": ["boolean"],
        "torsion_atom_count": [Interval(Integral, 2, None, closed="left")],
        "count_simulation": ["boolean"],
    }

    def __init__(
        self,
        fp_size: int = 2048,
        include_chirality: bool = False,
        torsion_atom_count: int = 4,
        count_simulation: bool = True,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.fp_size = fp_size
        self.include_chirality = include_chirality
        self.torsion_atom_count = torsion_atom_count
        self.count_simulation = count_simulation

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdFingerprintGenerator import GetTopologicalTorsionGenerator

        X = ensure_mols(X)

        gen = GetTopologicalTorsionGenerator(
            includeChirality=self.include_chirality,
            torsionAtomCount=self.torsion_atom_count,
            countSimulation=self.count_simulation,
            fpSize=self.fp_size,
        )

        if self.count:
            X = [gen.GetCountFingerprintAsNumPy(mol) for mol in X]
        else:
            X = [gen.GetFingerprintAsNumPy(mol) for mol in X]

        dtype = np.uint32 if self.count else np.uint8
        if self.sparse:
            return csr_array(X, dtype=dtype)
        else:
            return np.array(X, dtype=dtype)
