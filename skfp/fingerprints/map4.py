import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from datasketch import MinHash
from rdkit.Chem import MolToSmiles, PathToSubmol
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import FindAtomEnvironmentOfRadiusN, GetDistanceMatrix
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer
from skfp.validators import ensure_mols

"""
Code inspired by the original work of the authors of the MAP4 Fingerprint:
https://github.com/reymond-group/map4
"""


class MAP4Fingerprint(FingerprintTransformer):
    """MAP4 fingerprint."""

    def __init__(
        self,
        fp_size: int = 1024,
        radius: int = 2,
        variant: str = "bit",
        sparse: bool = False,
        count: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: int = 0,
    ):
        if variant not in ["bit", "count", "raw_hashes"]:
            raise ValueError("Variant must be one of: 'bit', 'count', 'raw_hashes'")

        super().__init__(
            n_features_out=fp_size,
            sparse=sparse,
            count=count,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
        self.fp_size = fp_size
        self.radius = radius
        self.variant = variant

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        X = ensure_mols(X)
        X = np.stack([self._calculate_single_mol_fingerprint(x) for x in X], dtype=int)

        if self.variant in ["bit", "count"]:
            X = np.mod(X, self.fp_size)
            X = np.stack([np.bincount(x, minlength=self.fp_size) for x in X])
            if self.variant == "bit":
                X = X > 0

        return csr_array(X) if self.sparse else np.array(X)

    def _calculate_single_mol_fingerprint(self, mol: Mol) -> np.ndarray:
        atoms_envs = self._get_atom_envs(mol)
        atom_env_pairs = self._get_atom_pair_shingles(mol, atoms_envs)
        encoder = MinHash(num_perm=self.fp_size, seed=self.random_state)
        encoder.update_batch(atom_env_pairs)
        fp = encoder.digest()
        return fp

    def _get_atom_envs(self, mol: Mol) -> Dict[int, List[Optional[str]]]:
        """
        For each atom get its environment, i.e. radius-hop neighborhood.
        """
        atoms_env = defaultdict(list)
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            atom_envs = [
                self._find_neighborhood(mol, idx, r) for r in range(1, self.radius + 1)
            ]
            atoms_env[idx].extend(atom_envs)

        return atoms_env

    def _find_neighborhood(
        self, mol: Mol, atom_idx: int, n_radius: int
    ) -> Optional[str]:
        """
        Get the radius-hop neighborhood for a given atom. If there is no neighborhood
        of a given radius, e.g. 2-hop neighborhood for [Li]F with just two atoms,
        returns None.
        """
        try:
            env = FindAtomEnvironmentOfRadiusN(mol, atom_idx, n_radius)
        except ValueError as e:
            # this error happens if radius is larger than possible
            if "bad atom index" in str(e):
                return None
            else:
                raise

        atom_map: Dict[int, int] = dict()

        submol = PathToSubmol(mol, env, atomMap=atom_map)

        if atom_idx in atom_map:
            return MolToSmiles(
                submol,
                rootedAtAtom=atom_map[atom_idx],
                canonical=True,
                isomericSmiles=False,
            )
        else:
            return None

    def _get_atom_pair_shingles(self, mol: Mol, atoms_envs: dict) -> List[bytes]:
        """
        Gets a list of atom-pair molecular shingles - circular structures written
        as SMILES, separated by the bond distance between the two atoms along the
        shortest path.
        """
        atom_pairs = []
        distance_matrix = GetDistanceMatrix(mol)
        num_atoms = mol.GetNumAtoms()
        shingle_dict: Dict[str, int] = defaultdict(int)

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

            for i in range(self.radius):
                env_a_radius = env_a[i]
                env_b_radius = env_b[i]

                # can be None if we couldn't get atom neighborhood of given radius
                if not env_a_radius or not env_b_radius:
                    continue

                ordered = sorted([env_a_radius, env_b_radius])
                shingle = f"{ordered[0]}|{dist}|{ordered[1]}"

                if self.count:
                    shingle_dict[shingle] += 1
                else:
                    atom_pairs.append(shingle.encode("utf-8"))

        if self.count:
            # shingle in format:
            # (radius i neighborhood of atom A) | (distance between atoms A and B) | \
            # (radius i neighborhood of atom B) | (shingle count)
            new_atom_pairs = [
                f"{shingle}|{shingle_count}".encode()
                for shingle, shingle_count in shingle_dict.items()
            ]
            atom_pairs.extend(new_atom_pairs)

        return atom_pairs
