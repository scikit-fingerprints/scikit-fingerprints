from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as spsparse

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
        include_chirality: bool = False,
        use_bond_types: bool = True,
        only_nonzero_invariants: bool = False,
        include_ring_membership: bool = True,
        count_bounds: Optional[List] = None,
        fp_size: int = 2048,
        n_jobs: int = 1,
        sparse: bool = False,
        count: bool = False,
        verbose: int = 0
    ):
        super().__init__(self, n_jobs, sparse, count)
        self.radius = radius
        self.include_chirality = include_chirality
        self.radius = radius
        self.include_chirality = include_chirality
        self.use_bond_types = use_bond_types
        self.only_nonzero_invariants = only_nonzero_invariants
        self.include_ring_membership = include_ring_membership
        self.count_bounds = count_bounds
        self.fp_size = fp_size

    def _get_generator(self):
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

        return GetMorganGenerator(
            radius=self.radius,
            includeChirality=self.include_chirality,
            useBondTypes=self.use_bond_types,
            onlyNonzeroInvariants=self.only_nonzero_invariants,
            includeRingMembership=self.include_ring_membership,
            countBounds=self.count_bounds,
            fpSize=self.fp_size,
        )

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        X = self._validate_input(X)
        return self._generate_fingerprints(X)

      
class AtomPairFingerprint(FingerprintTransformer):
    def __init__(
        self,
        min_distance: int = 1,
        max_distance: int = 30,
        include_chirality: bool = False,
        use_2D: bool = True,
        count_simulation: bool = True,
        count_bounds: Optional[List] = None,
        fp_size: int = 2048,
        n_jobs: int = 1,
        sparse: bool = False,
        count: bool = False,
        verbose: int = 0
):
        super().__init__(self, n_jobs=n_jobs, sparse=sparse, count=count, verbose=verbose)
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.include_chirality = include_chirality
        self.use_2D = use_2D
        self.count_simulation = count_simulation
        self.count_bounds = count_bounds
        self.fp_size = fp_size

    def _get_generator(self):
        from rdkit.Chem.rdFingerprintGenerator import GetAtomPairGenerator

        return GetAtomPairGenerator(
            minDistance=self.min_distance,
            maxDistance=self.max_distance,
            includeChirality=self.include_chirality,
            use2D=self.use_2D,
            countSimulation=self.count_simulation,
            countBounds=self.count_bounds,
            fpSize=self.fp_size,
        )

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        X = self._validate_input(X)
        return self._generate_fingerprints(X)


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
        verbose: int = 0
    ):
        super().__init__(self, n_jobs=n_jobs, sparse=sparse, count=count, verbose=verbose)
        self.include_chirality = include_chirality
        self.torsion_atom_count = torsion_atom_count
        self.count_simulation = count_simulation
        self.count_bounds = count_bounds
        self.fp_size = fp_size
        self.atom_invariants_generator = atom_invariants_generator

    def _get_generator(self):
        from rdkit.Chem.rdFingerprintGenerator import (
            GetTopologicalTorsionGenerator,
        )

        return GetTopologicalTorsionGenerator(
            includeChirality=self.include_chirality,
            torsionAtomCount=self.torsion_atom_count,
            countSimulation=self.count_simulation,
            countBounds=self.count_bounds,
            atomInvariantsGenerator=self.atom_invariants_generator,
            fpSize=self.fp_size,
        )

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        X = self._validate_input(X)
        return self._generate_fingerprints(X)


class MACCSKeysFingerprint(FingerprintTransformer):
    def __init__(self, 
                 sparse: bool = False, 
                 n_jobs: int = 1,
                 verbose: int = 0
                ):
        super().__init__(n_jobs=n_jobs, sparse=sparse, verbose=verbose)

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        X = self._validate_input(X)
        from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint

        X = [GetMACCSKeysFingerprint(x) for x in X]
        if self.sparse:
            return spsparse.csr_array(X)
        else:
            return np.array(X)


class ERGFingerprint(FingerprintTransformer):
    def __init__(
        self,
        atom_types: int = 0,
        fuzz_increment: float = 0.3,
        min_path: int = 1,
        max_path: int = 15,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(n_jobs=n_jobs, sparse=sparse, verbose=verbose)
        self.atom_types = atom_types
        self.fuzz_increment = fuzz_increment
        self.min_path = min_path
        self.max_path = max_path

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        X = self._validate_input(X)
        from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

        X = [
            GetErGFingerprint(
                x,
                atomTypes=self.atom_types,
                fuzzIncrement=self.fuzz_increment,
                minPath=self.min_path,
                maxPath=self.max_path,
            )
            for x in X
        ]
        
        if self.sparse:
            return spsparse.csr_array(X)
        else:
            return np.array(X)
          

class MAP4Fingerprint(FingerprintTransformer):
    def __init__(
        self,
        dimensions: int = 1024,
        radius: int = 2,
        is_counted: bool = False,
        is_folded: bool = False,
        random_state: int = 0,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(n_jobs=n_jobs, verbose=verbose)
        self.dimensions = dimensions
        self.radius = radius
        self.is_counted = is_counted
        self.random_state = random_state

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        from featurizers.map4_mhfp_helper_functions import (
            get_map4_fingerprint,
        )

        fp_args = {
            "dimensions": self.dimensions,
            "radius": self.radius,
            "is_counted": self.is_counted,
            "random_state": self.random_state,
        }

        return np.array([get_map4_fingerprint(x, **fp_args) for x in X])


class MHFP(FingerprintTransformer):
    def __init__(
        self,
        dimensions: int = 1024,
        radius: int = 2,
        is_counted: bool = False,
        random_state: int = 0,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(n_jobs=n_jobs, verbose=verbose)
        self.dimensions = dimensions
        self.radius = radius
        self.is_counted = is_counted
        self.random_state = random_state

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        from featurizers.map4_mhfp_helper_functions import get_mhfp

        fp_args = {
            "dimensions": self.dimensions,
            "radius": self.radius,
            "is_counted": self.is_counted,
            "random_state": self.random_state,
        }

        return np.array([get_mhfp(x, **fp_args) for x in X])
