"""
Code inspired by the original work of the authors of the MAP4 Fingerprint:
https://github.com/reymond-group/map4
"""

import itertools
from collections import defaultdict

import numpy as np
from rdkit.Chem import MolToSmiles, PathToSubmol
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import FindAtomEnvironmentOfRadiusN, GetDistanceMatrix


def _find_neighborhood(Mol: Mol, idx: int, radius: int) -> str:
    """
    Function for getting the neighborhood for given atom. By the neighborhood, we mean the structures,
    that are adjacent to the given atom.
    """
    env = FindAtomEnvironmentOfRadiusN(Mol, idx, radius)
    atom_map = {}

    submol = PathToSubmol(Mol, env, atomMap=atom_map)

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


def _get_atom_envs(Mol: Mol, radius: int):
    """
    For each atom get its environment.
    """
    atoms_env = defaultdict(lambda: [])
    for atom in Mol.GetAtoms():
        idx = atom.GetIdx()
        for r in range(1, radius + 1):
            atoms_env[idx].append(_find_neighborhood(Mol, idx, r))

    return atoms_env


def _all_pairs(Mol: Mol, atoms_envs: dict, radius: int, is_counted: bool):
    """
    Gets a list of atom-pair molecular shingles - circular structures written as SMILES, separated by the bond distance
    between the two atoms along the shortest path.
    """
    atom_pairs = []
    distance_matrix = GetDistanceMatrix(Mol)
    num_atoms = Mol.GetNumAtoms()
    shingle_dict = defaultdict(int)

    # Iterate through all pairs of atoms and radius. Shingles are stored in format:
    # (neighborhood of atom A of radius i) | (distance between atom A and atom B) | (neighborhood of atom B of radius i)
    #
    # If we want to count the shingles (is_counted), we increment their value in shingle_dict.
    # After nested for-loop, all shingles, with their respective counts, will be added to atom_pairs list.
    for idx_1, idx_2 in itertools.combinations(range(num_atoms), 2):
        # distance_matrix consists of floats as integers, so they need to be converted to integers first
        dist = str(int(distance_matrix[idx_1][idx_2]))

        for i in range(radius):
            env_a = atoms_envs[idx_1][i]
            env_b = atoms_envs[idx_2][i]

            if not len(env_a) or not len(env_b):
                continue

            ordered = sorted([env_a, env_b])

            shingle = f"{ordered[0]}|{dist}|{ordered[1]}"

            if is_counted:
                shingle_dict[shingle] += 1
            else:
                atom_pairs.append(shingle.encode("utf-8"))

    if is_counted:
        # Shingle in a format:
        # (neighborhood of atom A of radius i) | (distance between atom A and atom B) | \
        # (neighborhood of atom B of radius i) | (shingle count)
        for shingle, shingle_val in shingle_dict.items():
            shingle += "|" + str(shingle_val)
            atom_pairs.append(shingle.encode("utf-8"))

    return atom_pairs


def get_map4_fingerprint(
    Mol: Mol,
    dimensions: int,
    radius: int,
    is_counted: bool,
    random_state: int,
):
    # TODO - There are certain molecules, for which this function will return a error:
    #   https://github.com/Arch4ngel21/emf/issues/13
    #   So for now it's handled by try/except
    try:
        from datasketch.minhash import MinHash

        atoms_envs = _get_atom_envs(Mol, radius)

        atom_env_pairs = _all_pairs(Mol, atoms_envs, radius, is_counted)

        encoder = MinHash(seed=random_state)
        for pair in atom_env_pairs:
            encoder.update(pair)

        return encoder.digest()
    except ValueError:
        return np.full(shape=dimensions, fill_value=-1)


def get_mhfp(
    Mol: Mol,
    dimensions: int,
    radius: int,
    is_counted: bool,
    random_state: int,
):
    try:
        from mhfp.encoder import MHFPEncoder

        atoms_envs = _get_atom_envs(Mol, radius)

        atom_env_pairs = _all_pairs(Mol, atoms_envs, radius, is_counted)

        encoder = MHFPEncoder(n_permutations=dimensions, seed=random_state)
        fp_hash = encoder.hash(set(atom_env_pairs))
        return encoder.fold(fp_hash, dimensions)
    except ValueError:
        return np.full(shape=dimensions, fill_value=-1)
