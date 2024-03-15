import itertools
from collections import defaultdict
from typing import Union, List

import numpy as np
import pandas as pd
from datasketch import MinHash
from rdkit.Chem import MolToSmiles, PathToSubmol
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import FindAtomEnvironmentOfRadiusN, GetDistanceMatrix
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer

"""
Code inspired by the original work of the authors of the MAP4 Fingerprint:
https://github.com/reymond-group/map4
"""


class MAP4Fingerprint(FingerprintTransformer):
    def __init__(
        self,
        fp_size: int = 1024,
        radius: int = 2,
        sparse: bool = False,
        count: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
        random_state: int = 0,
    ):
        super().__init__(
            sparse=sparse,
            count=count,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
        self.dimensions = fp_size
        self.radius = radius

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, List[str]]
    ) -> Union[np.ndarray, csr_array]:
        X = self._validate_input(X)
        X = [
            self._calculate_single_mol_fingerprint(
                x,
                dimensions=self.dimensions,
                radius=self.radius,
                count=self.count,
                random_state=self.random_state,
            )
            for x in X
        ]

        if self.sparse:
            return csr_array(np.stack(X))
        else:
            return np.stack(X)

    def _calculate_single_mol_fingerprint(
        self,
        mol: Mol,
        dimensions: int = 1024,
        radius: int = 2,
        count: bool = False,
        random_state: int = 0,
    ):
        # TODO: does not work for some molecules, for now handled by try/except
        try:
            atoms_envs = self._get_atom_envs(mol, radius)
            atom_env_pairs = self._get_atom_pair_shingles(
                mol, atoms_envs, radius, count
            )
            encoder = MinHash(num_perm=dimensions, seed=random_state)
            encoder.update_batch(atom_env_pairs)
            return encoder.digest()
        except ValueError:
            return np.full(shape=dimensions, fill_value=-1)

    def _find_neighborhood(self, mol: Mol, idx: int, radius: int) -> str:
        """
        Function for getting the neighborhood for given atom, i.e. structures
        adjacent to it.
        """
        env = FindAtomEnvironmentOfRadiusN(mol, idx, radius)
        atom_map = {}

        submol = PathToSubmol(mol, env, atomMap=atom_map)

        if idx in atom_map:
            smiles = MolToSmiles(
                submol,
                rootedAtAtom=atom_map[idx],
                canonical=True,
                isomericSmiles=False,
            )
            return smiles
        else:
            return ""

    def _get_atom_envs(self, mol: Mol, radius: int) -> dict:
        """
        For each atom get its environment, i.e. radius-hop neighborhood.
        """
        atoms_env = defaultdict(list)
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            new_values = [
                self._find_neighborhood(mol, idx, r)
                for r in range(1, radius + 1)
            ]
            atoms_env[idx].extend(new_values)

        return atoms_env

    def _get_atom_pair_shingles(
        self, mol: Mol, atoms_envs: dict, radius: int, count: bool
    ) -> List[str]:
        """
        Gets a list of atom-pair molecular shingles - circular structures written
        as SMILES, separated by the bond distance between the two atoms along the
        shortest path.
        """
        atom_pairs = []
        distance_matrix = GetDistanceMatrix(mol)
        num_atoms = mol.GetNumAtoms()
        shingle_dict = defaultdict(int)

        # Iterate through all pairs of atoms and radius. Shingles are stored in format:
        # (radius i neighborhood of atom A) | (distance between atoms A and B) | (radius i neighborhood of atom B)
        #
        # If we want to count the shingles, we increment their value in shingle_dict
        # After nested for-loop, all shingles, with their respective counts, will be added to atom_pairs list.
        for idx_1, idx_2 in itertools.combinations(range(num_atoms), 2):
            # distance_matrix consists of floats as integers, so they need to be converted to integers first
            dist = str(int(distance_matrix[idx_1][idx_2]))
            env_a = atoms_envs[idx_1]
            env_b = atoms_envs[idx_2]

            for i in range(radius):
                env_a_radius = env_a[i]
                env_b_radius = env_b[i]

                if not len(env_a_radius) or not len(env_b_radius):
                    continue

                ordered = sorted([env_a_radius, env_b_radius])
                shingle = f"{ordered[0]}|{dist}|{ordered[1]}"

                if count:
                    shingle_dict[shingle] += 1
                else:
                    atom_pairs.append(shingle.encode("utf-8"))

        if count:
            # shingle in format:
            # (radius i neighborhood of atom A) | (distance between atoms A and B) | \
            # (radius i neighborhood of atom B) | (shingle count)
            new_atom_pairs = [
                f"{shingle}|{shingle_count}".encode("utf-8")
                for shingle, shingle_count in shingle_dict.items()
            ]
            atom_pairs.extend(new_atom_pairs)

        return atom_pairs
