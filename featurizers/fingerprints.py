import logging
from typing import Union

import e3fp.fingerprint.fprint
import numpy as np
import pandas as pd
from rdkit.Chem import MolToSmiles

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
        radius: int = 2,
        n_bits: int = 2048,
        use_chirality: bool = False,
        use_bond_types: bool = True,
        use_features: bool = False,
        sparse: bool = False,
        result_type: str = "default",
        n_jobs: int = 1,
    ):
        assert result_type in ["default", "as_bit_vect", "hashed"]

        super().__init__(n_jobs)
        self.radius = radius
        self.n_bits = n_bits
        self.use_chirality = use_chirality
        self.use_bond_types = use_bond_types
        self.use_features = use_features
        self.result_type = result_type

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        if self.result_type == "default":
            from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint

            return np.array(
                [
                    GetMorganFingerprint(
                        x,
                        radius=self.radius,
                        useChirality=self.use_chirality,
                        useBondTypes=self.use_bond_types,
                        useFeatures=self.use_features,
                    )
                    for x in X
                ]
            )
        elif self.result_type == "as_bit_vect":
            from rdkit.Chem.rdMolDescriptors import (
                GetMorganFingerprintAsBitVect,
            )

            return np.array(
                [
                    GetMorganFingerprintAsBitVect(
                        x,
                        radius=self.radius,
                        useChirality=self.use_chirality,
                        useBondTypes=self.use_bond_types,
                        useFeatures=self.use_features,
                        nBits=self.n_bits,
                    )
                    for x in X
                ]
            )
        else:
            from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint

            return np.array(
                [
                    GetHashedMorganFingerprint(
                        x,
                        radius=self.radius,
                        useChirality=self.use_chirality,
                        useBondTypes=self.use_bond_types,
                        useFeatures=self.use_features,
                        nBits=self.n_bits,
                    )
                    for x in X
                ]
            )


class MACCSKeysFingerprint(FingerprintTransformer):
    def __init__(
        self, sparse: bool = False, n_jobs: int = 1
    ):  # the sparse parameter will be unused
        super().__init__(n_jobs)

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint

        return np.array([GetMACCSKeysFingerprint(x) for x in X])


class AtomPairFingerprint(FingerprintTransformer):
    def __init__(
        self,
        n_bits: int = 2048,
        min_length: int = 1,
        max_length: int = 30,
        from_atoms: int = 0,
        ignore_atoms: int = 0,
        atom_invariants: int = 0,
        n_bits_per_entry: int = 4,
        include_chirality: bool = False,
        use_2D: bool = True,
        sparse: bool = False,
        result_type: str = "default",
        n_jobs: int = 1,
    ):
        assert result_type in ["default", "as_bit_vect", "hashed"]

        super().__init__(n_jobs)
        self.n_bits = n_bits
        self.min_length = min_length
        self.max_length = max_length
        self.from_atoms = from_atoms
        self.ignore_atoms = ignore_atoms
        self.atom_invariants = atom_invariants
        self.n_bits_per_entry = n_bits_per_entry
        self.include_chirality = include_chirality
        self.use_2D = use_2D
        self.result_type = result_type

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        fp_args = {
            "minLength": self.min_length,
            "maxLength": self.max_length,
            "fromAtoms": self.from_atoms,
            "ignoreAtoms": self.ignore_atoms,
            "atomInvariants": self.atom_invariants,
            "includeChirality": self.include_chirality,
            "use2D": self.use_2D,
        }

        if self.result_type == "default":
            from rdkit.Chem.rdMolDescriptors import GetAtomPairFingerprint

            fp_function = GetAtomPairFingerprint
        elif self.result_type == "as_bit_vect":
            from rdkit.Chem.rdMolDescriptors import (
                GetHashedAtomPairFingerprintAsBitVect,
            )

            fp_function = GetHashedAtomPairFingerprintAsBitVect
            fp_args["nBits"] = self.n_bits
            fp_args["nBitsPerEntry"] = self.n_bits_per_entry
        else:
            from rdkit.Chem.rdMolDescriptors import (
                GetHashedAtomPairFingerprint,
            )

            fp_function = GetHashedAtomPairFingerprint
            fp_args["nBits"] = self.n_bits

        return np.array([fp_function(x, **fp_args) for x in X])


class TopologicalTorsionFingerprint(FingerprintTransformer):
    def __init__(
        self,
        n_bits: int = 2048,
        target_size: int = 4,
        from_atoms: int = 0,
        ignore_atoms: int = 0,
        atom_invariants: int = 0,
        include_chirality: bool = False,
        sparse: bool = False,
        result_type: str = "default",
        n_jobs: int = 1,
    ):
        assert result_type in ["default", "as_bit_vect", "hashed"]

        super().__init__(n_jobs)
        self.n_bits = n_bits
        self.target_size = target_size
        self.from_atoms = from_atoms
        self.ignore_atoms = ignore_atoms
        self.atom_invariants = atom_invariants
        self.include_chirality = include_chirality
        self.result_type = result_type

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        fp_args = {
            "targetSize": self.target_size,
            "fromAtoms": self.from_atoms,
            "ignoreAtoms": self.ignore_atoms,
            "atomInvariants": self.atom_invariants,
            "includeChirality": self.include_chirality,
        }

        if self.result_type == "default":
            from rdkit.Chem.rdMolDescriptors import (
                GetTopologicalTorsionFingerprint,
            )

            fp_function = GetTopologicalTorsionFingerprint
        elif self.result_type == "as_bit_vect":
            from rdkit.Chem.rdMolDescriptors import (
                GetHashedTopologicalTorsionFingerprintAsBitVect,
            )

            fp_function = GetHashedTopologicalTorsionFingerprintAsBitVect
            fp_args["nBits"] = self.n_bits
        else:
            from rdkit.Chem.rdMolDescriptors import (
                GetHashedTopologicalTorsionFingerprint,
            )

            fp_function = GetHashedTopologicalTorsionFingerprint
            fp_args["nBits"] = self.n_bits

        return np.array([fp_function(x, **fp_args) for x in X])


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
        super().__init__(n_jobs)
        self.atom_types = atom_types
        self.fuzz_increment = fuzz_increment
        self.min_path = min_path
        self.max_path = max_path

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

        fp_args = {
            "atomTypes": self.atom_types,
            "fuzzIncrement": self.fuzz_increment,
            "minPath": self.min_path,
            "maxPath": self.max_path,
        }

        return np.array([GetErGFingerprint(x, **fp_args) for x in X])


class MAP4Fingerprint(FingerprintTransformer):
    def __init__(
        self,
        dimensions: int = 1024,
        radius: int = 2,
        is_counted: bool = False,
        is_folded: bool = False,
        return_strings: bool = False,
        random_state: int = 0,
        n_jobs: int = 1,
    ):
        super().__init__(n_jobs)
        self.dimensions = dimensions
        self.radius = radius
        self.is_counted = is_counted
        self.is_folded = is_folded
        self.return_strings = return_strings
        self.random_state = random_state

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        from featurizers.map4 import GetMAP4Fingerprint

        fp_args = {
            "dimensions": self.dimensions,
            "radius": self.radius,
            "is_counted": self.is_counted,
            "is_folded": self.is_folded,
            "return_strings": self.return_strings,
            "random_state": self.random_state,
        }

        return np.array([GetMAP4Fingerprint(x, **fp_args) for x in X])


class E3FP(FingerprintTransformer):
    def __init__(
        self,
        confgen_params: dict,
        fprint_params: dict,
        is_folded: bool = False,
        fold_bits: int = 1024,
        standardise: bool = False,
        random_state: int = 0,
        n_jobs: int = 1,
        verbose: int = 0,
    ):
        if "first" not in confgen_params:
            confgen_params["first"] = fprint_params.get("first", 1)
        if "first" not in fprint_params:
            fprint_params["first"] = confgen_params.get("first", 1)

        # Get necessary default values if not specified
        from e3fp.conformer.generate import (
            NUM_CONF_DEF,
            POOL_MULTIPLIER_DEF,
            RMSD_CUTOFF_DEF,
            MAX_ENERGY_DIFF_DEF,
            FORCEFIELD_DEF,
            SEED_DEF,
        )

        confgen_params["num_conf"] = confgen_params.get(
            "num_conf", NUM_CONF_DEF
        )
        confgen_params["pool_multiplier"] = confgen_params.get(
            "pool_multiplier", POOL_MULTIPLIER_DEF
        )
        confgen_params["rmsd_cutoff"] = confgen_params.get(
            "rmsd_cutoff", RMSD_CUTOFF_DEF
        )
        confgen_params["max_energy_diff"] = confgen_params.get(
            "max_energy_diff", MAX_ENERGY_DIFF_DEF
        )
        confgen_params["forcefield"] = confgen_params.get(
            "forcefield", FORCEFIELD_DEF
        )
        confgen_params["seed"] = confgen_params.get("seed", SEED_DEF)
        confgen_params["get_values"] = True

        super().__init__(n_jobs)
        self.confgen_params = confgen_params
        self.fprint_params = fprint_params
        self.is_folded = is_folded
        self.fold_bits = fold_bits
        self.standardise = standardise
        self.first = confgen_params.get("first", 1)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        from e3fp.conformer.util import mol_from_smiles
        from e3fp.conformer.generator import ConformerGenerator
        from e3fp.pipeline import fprints_from_mol

        # Disable logging
        if self.verbose == 0:
            logging.getLogger("e3fp").setLevel(logging.CRITICAL)

        conf_gen = ConformerGenerator(
            **self.confgen_params,
        )

        result = []

        for x in X:
            input_mol = mol_from_smiles(
                smiles=MolToSmiles(x),
                name=MolToSmiles(x),
                standardise=self.standardise,
            )

            # Generating conformers. Only few first conformers with lowest energy are used - specified by self.first
            mol, values = conf_gen.generate_conformers(input_mol)

            max_conformers, indices, energies, rmsds_mat = values

            fps = fprints_from_mol(mol, fprint_params=self.fprint_params)

            # Set a property for each fingerprint, to be able to obtain energy later
            for i in range(len(fps)):
                fps[i].set_prop("Energy", energies[i])

            if self.is_folded:
                for i in range(len(fps)):
                    fps[i] = fps[i].fold(self.fold_bits)

            result.extend(fps)

        result = np.array(result, dtype=object)

        return result
