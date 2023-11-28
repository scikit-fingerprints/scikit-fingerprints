from typing import List, Optional, Union

import e3fp.fingerprint.fprint
import numpy as np
import pandas as pd
import scipy.sparse as spsparse
from e3fp.conformer.generate import (
    FORCEFIELD_DEF,
    MAX_ENERGY_DIFF_DEF,
    NUM_CONF_DEF,
    POOL_MULTIPLIER_DEF,
    RMSD_CUTOFF_DEF,
)
from e3fp.conformer.util import mol_to_standardised_mol
from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles
from rdkit.Chem.PropertyMol import PropertyMol

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
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
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
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
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
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        X = self._validate_input(X)
        return self._generate_fingerprints(X)


class MACCSKeysFingerprint(FingerprintTransformer):
    def __init__(
        self,
        sparse: bool = False,
        n_jobs: int = 1,
        verbose: int = 0,
        random_state: int = 0,
        count: bool = False,  # unused
    ):
        super().__init__(
            n_jobs=n_jobs,
            sparse=sparse,
            verbose=verbose,
            random_state=random_state,
            count=count,
        )

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
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
        random_state: int = 0,
        count: bool = False,  # unused
    ):
        super().__init__(
            n_jobs=n_jobs,
            sparse=sparse,
            verbose=verbose,
            random_state=random_state,
            count=count,
        )
        self.atom_types = atom_types
        self.fuzz_increment = fuzz_increment
        self.min_path = min_path
        self.max_path = max_path

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
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
        random_state: int = 0,
        sparse: bool = False,
        count: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            sparse=sparse,
            count=count,
            verbose=verbose,
            random_state=random_state,
        )
        self.dimensions = dimensions
        self.radius = radius

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        X = self._validate_input(X)
        from featurizers.map4_mhfp_helper_functions import get_map4_fingerprint

        X = [
            get_map4_fingerprint(
                x,
                dimensions=self.dimensions,
                radius=self.radius,
                count=self.count,
                random_state=self.random_state,
            )
            for x in X
        ]

        if self.sparse:
            return spsparse.csr_array(np.stack(X))
        else:
            return np.stack(X)


class MHFP(FingerprintTransformer):
    def __init__(
        self,
        dimensions: int = 1024,
        radius: int = 2,
        random_state: int = 0,
        sparse: bool = False,
        count: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            sparse=sparse,
            count=count,
            verbose=verbose,
            random_state=random_state,
        )
        self.dimensions = dimensions
        self.radius = radius

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        X = self._validate_input(X)
        from featurizers.map4_mhfp_helper_functions import get_mhfp

        X = [
            get_mhfp(
                x,
                dimensions=self.dimensions,
                radius=self.radius,
                random_state=self.random_state,
                count=self.count,
            )
            for x in X
        ]
        if self.sparse:
            return spsparse.csr_array(X)
        else:
            return np.array(X)


class E3FP(FingerprintTransformer):
    def __init__(
        self,
        bits: int = 4090,
        radius_multiplier: float = 1.5,
        rdkit_invariants: bool = True,
        first: int = 1,
        num_conf: int = NUM_CONF_DEF,
        pool_multiplier: float = POOL_MULTIPLIER_DEF,
        rmsd_cutoff: float = RMSD_CUTOFF_DEF,
        max_energy_diff: float = MAX_ENERGY_DIFF_DEF,
        force_field: float = FORCEFIELD_DEF,
        get_values: bool = True,
        is_folded: bool = False,
        fold_bits: int = 1024,
        sparse: bool = False,
        n_jobs: int = 1,
        verbose: int = 0,
        random_state: int = 0,
        aggregation_type: str = "min_energy",
        count: bool = False,  # unused
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
            sparse=sparse,
            random_state=random_state,
            count=count,
        )
        self.bits = bits
        self.radius_multiplier = radius_multiplier
        self.rdkit_invariants = rdkit_invariants
        self.first = first
        self.num_conf = num_conf
        self.pool_multiplier = pool_multiplier
        self.rmsd_cutoff = rmsd_cutoff
        self.max_energy_diff = max_energy_diff
        self.force_field = force_field
        self.get_values = get_values
        self.is_folded = is_folded
        self.fold_bits = fold_bits
        self.aggregation_type = aggregation_type

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        from e3fp.conformer.generator import ConformerGenerator
        from e3fp.pipeline import fprints_from_mol

        conf_gen = ConformerGenerator(
            first=self.first,
            num_conf=self.num_conf,
            pool_multiplier=self.pool_multiplier,
            rmsd_cutoff=self.rmsd_cutoff,
            max_energy_diff=self.max_energy_diff,
            forcefield=self.force_field,
            get_values=self.get_values,
            seed=self.random_state,
        )

        result = []

        for x in X:
            if isinstance(x, Mol):
                smiles = MolToSmiles(x)
                mol = x
            else:
                smiles = x
                mol = MolFromSmiles(x)

            mol.SetProp("_Name", smiles)
            mol = PropertyMol(mol)
            mol.SetProp("_SMILES", smiles)

            # Generating conformers. Only few first conformers with lowest energy are used - specified by self.first
            mol, values = conf_gen.generate_conformers(mol)
            fps = fprints_from_mol(
                mol,
                fprint_params={
                    "bits": self.bits,
                    "radius_multiplier": self.radius_multiplier,
                    "rdkit_invariants": self.rdkit_invariants,
                },
            )

            # TODO: in future - add other aggregation types
            if self.aggregation_type == "min_energy":
                energies = values[2]
                fp = fps[np.argmin(energies)]
            else:
                fp = fps[0]

            if self.is_folded:
                fp = fp.fold(self.fold_bits)

            result.append(fp.to_vector())

        if self.sparse:
            return spsparse.vstack(result)
        else:
            return np.array([fp.toarray().squeeze() for fp in result])
