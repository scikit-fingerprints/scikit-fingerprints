from typing import List, Optional, Union

import numpy as np
import pandas as pd
from rdkit.Chem.rdMolDescriptors import AtomPairsParameters

from base import FingerprintTransformer

"""
If during multiprocessing occurs MaybeEncodingError, first check if there isn't thrown any exception inside
worker function! (That error isn't very informative and this tip might save you a lot of time)
"""

"""
fp_descriptors needs to be here or inside _transform() of a specific class (cannot be defined inside the __init__() of
that class), otherwise pickle gets angry:
        TypeError: cannot pickle 'Boost.Python.function' object
"""


class MorganFingerprint(FingerprintTransformer):
    def __init__(
        self,
        radius: int = 3,
        includeChirality: bool = False,
        useBondTypes: bool = True,
        onlyNonzeroInvariants: bool = False,
        includeRingMembership: bool = True,
        countBounds: Optional[List] = None,
        fpSize: int = 2048,
        n_jobs: int = 1,
        sparse: bool = False,
        count: bool = False,
    ):
        FingerprintTransformer.__init__(self, n_jobs, sparse, count)

        self.fp_generator_kwargs = {
            "radius": radius,
            "includeChirality": includeChirality,
            "useBondTypes": useBondTypes,
            "onlyNonzeroInvariants": onlyNonzeroInvariants,
            "includeRingMembership": includeRingMembership,
            "countBounds": countBounds,
            "fpSize": fpSize,
        }

    def _get_generator(self):
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

        return GetMorganGenerator(**self.fp_generator_kwargs)

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        X = self._validate_input(X)
        return self._generate_fingerprints(X)


class MACCSKeysFingerprint(FingerprintTransformer):
    def __init__(
        self, sparse: bool = False, n_jobs: int = 1
    ):  # the sparse parameter will be unused
        super().__init__(n_jobs=n_jobs, sparse=sparse)

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        X = self._validate_input(X)
        from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint

        return np.array([GetMACCSKeysFingerprint(x) for x in X])


class AtomPairFingerprint(FingerprintTransformer):
    def __init__(
        self,
        minDistance: int = 1,
        maxDistance: int = 30,
        includeChirality: bool = False,
        use2D: bool = True,
        countSimulation: bool = True,
        countBounds: Optional[List] = None,
        fpSize: int = 2048,
        n_jobs: int = 1,
        sparse: bool = False,
        count: bool = False,
    ):
        FingerprintTransformer.__init__(self, n_jobs, sparse, count)

        self.fp_generator_kwargs = {
            "minDistance": minDistance,
            "maxDistance": maxDistance,
            "includeChirality": includeChirality,
            "use2D": use2D,
            "countSimulation": countSimulation,
            "countBounds": countBounds,
            "fpSize": fpSize,
        }

    def _get_generator(self):
        from rdkit.Chem.rdFingerprintGenerator import GetAtomPairGenerator

        return GetAtomPairGenerator(**self.fp_generator_kwargs)

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        X = self._validate_input(X)
        return self._generate_fingerprints(X)


class TopologicalTorsionFingerprint(FingerprintTransformer):
    def __init__(
        self,
        includeChirality: bool = False,
        torsionAtomCount: int = 4,
        countSimulation: bool = True,
        countBounds: Optional[List] = None,
        fpSize: int = 2048,
        atomInvariantsGenerator: Optional[List] = None,
        n_jobs: int = 1,
        sparse: bool = False,
        count: bool = False,
    ):
        FingerprintTransformer.__init__(self, n_jobs, sparse, count)

        self.fp_generator_kwargs = {
            "includeChirality": includeChirality,
            "torsionAtomCount": torsionAtomCount,
            "countSimulation": countSimulation,
            "countBounds": countBounds,
            "atomInvariantsGenerator": atomInvariantsGenerator,
            "fpSize": fpSize,
        }

    def _get_generator(self):
        from rdkit.Chem.rdFingerprintGenerator import (
            GetTopologicalTorsionGenerator,
        )

        return GetTopologicalTorsionGenerator(**self.fp_generator_kwargs)

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        X = self._validate_input(X)
        return self._generate_fingerprints(X)


class ERGFingerprint(FingerprintTransformer):
    def __init__(
        self,
        atom_types: int = 0,
        fuzz_increment: float = 0.3,
        min_path: int = 1,
        max_path: int = 15,
        sparse: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(n_jobs, sparse)
        self.atom_types = atom_types
        self.fuzz_increment = fuzz_increment
        self.min_path = min_path
        self.max_path = max_path

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        X = self._validate_input(X)
        from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

        fp_args = {
            "atomTypes": self.atom_types,
            "fuzzIncrement": self.fuzz_increment,
            "minPath": self.min_path,
            "maxPath": self.max_path,
        }

        return np.array([GetErGFingerprint(x, **fp_args) for x in X])
