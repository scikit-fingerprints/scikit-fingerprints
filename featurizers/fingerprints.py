import multiprocessing as mp
from typing import Union

import numpy as np
import pandas as pd
import rdkit.Chem as Chem

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
        result_type: str = "default",
        n_jobs: int = 1,
    ):
        assert result_type in ["default", "as_bit_vect", "hashed"]

        super().__init__(n_jobs)

        self.fp_args = {
            "radius": radius,
            "useChirality": use_chirality,
            "useBondTypes": use_bond_types,
            "useFeatures": use_features,
        }

        if result_type in ["as_bit_vect", "hashed"]:
            self.fp_args["nBits"] = n_bits

        self.fp_descriptor_name = "morgan_" + result_type


class MACCSKeysFingerprint(FingerprintTransformer):
    def __init__(self, n_jobs: int = 1):
        super().__init__(n_jobs)
        self.fp_args = {}
        self.fp_descriptor_name = "maccs_keys"


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
        result_type: str = "default",
        n_jobs: int = 1,
    ):
        assert result_type in ["default", "as_bit_vect", "hashed"]

        super().__init__(n_jobs)
        self.fp_args = {
            "minLength": min_length,
            "maxLength": max_length,
            "fromAtoms": from_atoms,
            "ignoreAtoms": ignore_atoms,
            "atomInvariants": atom_invariants,
            "includeChirality": include_chirality,
            "use2D": use_2D,
        }

        if result_type == "hashed":
            self.fp_args["nBits"] = n_bits
        elif result_type == "as_bit_vect":
            self.fp_args["nBits"] = n_bits
            self.fp_args["nBitsPerEntry"] = n_bits_per_entry

        self.fp_descriptor_name = "atom_pair_" + result_type


class TopologicalTorsionFingerprint(FingerprintTransformer):
    def __init__(
        self,
        n_bits: int = 2048,
        target_size: int = 4,
        from_atoms: int = 0,
        ignore_atoms: int = 0,
        atom_invariants: int = 0,
        include_chirality: bool = False,
        result_type: str = "default",
        n_jobs: int = 1,
    ):
        assert result_type in ["default", "as_bit_vect", "hashed"]

        super().__init__(n_jobs)
        self.fp_args = {
            "targetSize": target_size,
            "fromAtoms": from_atoms,
            "ignoreAtoms": ignore_atoms,
            "atomInvariants": atom_invariants,
            "includeChirality": include_chirality,
        }

        if result_type in ["hashed", "as_bit_vect"]:
            self.fp_args["nBits"] = n_bits

        self.fp_descriptor_name = "topological_torsion_" + result_type


class ERGFingerprint(FingerprintTransformer):
    def __init__(
        self,
        atom_types: int = 0,
        fuzz_increment: float = 0.3,
        min_path: int = 1,
        max_path: int = 15,
        n_jobs: int = 1,
    ):
        super().__init__(n_jobs)

        self.fp_args = {
            "atomTypes": atom_types,
            "fuzzIncrement": fuzz_increment,
            "minPath": min_path,
            "maxPath": max_path,
        }

        self.fp_descriptor_name = "erg"
