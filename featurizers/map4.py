"""
Code inspired by the original work of the authors of the MAP4 Fingerprint:
https://github.com/reymond-group/map4
"""

import itertools
from collections import defaultdict

import numpy as np
from mhfp.encoder import MHFPEncoder
from rdkit.Chem import MolToSmiles, PathToSubmol
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import FindAtomEnvironmentOfRadiusN, GetDistanceMatrix

from featurizers.minhash import Minhash


def _find_env(Mol: Mol, idx: int, radius: int) -> str:
    """
    Function for getting the environment for given atom. By the environment, we mean the structures,
    that are adjacent to the given atom.

    :param Mol: - given molecule
    :param idx: - index of the heavy atom in the molecule of which an environment will be found
    :param radius: - maximum radius of search from the given atom
    :return: string in SMILES format
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
    return ""


def _get_atom_envs(Mol: Mol, radius: int):
    """
    For each atom get its environment.

    :param Mol:
    :param radius:
    :return:
    """
    atoms_env = {}
    for atom in Mol.GetAtoms():
        idx = atom.GetIdx()
        for r in range(1, radius + 1):
            if idx not in atoms_env:
                atoms_env[idx] = []
            atoms_env[idx].append(_find_env(Mol, idx, r))
    return atoms_env


def _all_pairs(Mol: Mol, atoms_envs: dict, radius: int, is_counted: bool):
    """
    Gets a list of atom-pair molecular shingles - circular structures written as SMILES, separated by the bond distance
    between the two atoms along the shortest path.

    :param Mol:
    :param atoms_envs:
    :param radius:
    :param is_counted:
    :return:
    """
    atom_pairs = []
    distance_matrix = GetDistanceMatrix(Mol)
    num_atoms = Mol.GetNumAtoms()
    shingle_dict = defaultdict(int)

    for idx_1, idx_2 in itertools.combinations(range(num_atoms), 2):
        dist = str(int(distance_matrix[idx_1][idx_2]))

        for i in range(radius):
            env_a = atoms_envs[idx_1][i]
            env_b = atoms_envs[idx_2][i]

            ordered = sorted([env_a, env_b])

            shingle = "{}|{}|{}".format(ordered[0], dist, ordered[1])

            if is_counted:
                shingle_dict[shingle] += 1
                shingle += "|" + str(shingle_dict[shingle])

            atom_pairs.append(shingle.encode("utf-8"))
    return list(set(atom_pairs))


def GetMAP4Fingerprint(
    Mol: Mol,
    dimensions: int = 1024,
    radius: int = 2,
    is_counted: bool = False,
    is_folded: bool = False,
    return_strings: bool = False,
    random_state: int = 0,
):
    # TODO - There are certain molecules, for which this function will return a error:
    #   https://github.com/Arch4ngel21/emf/issues/13
    #   So for now it's handled by try/except
    try:
        if is_folded:
            encoder = MHFPEncoder(n_permutations=dimensions, seed=random_state)
        else:
            encoder = Minhash(dimensions)

        atoms_envs = _get_atom_envs(Mol, radius)

        atom_env_pairs = _all_pairs(Mol, atoms_envs, radius, is_counted)

        if is_folded:
            fp_hash = encoder.hash(set(atom_env_pairs))
            return encoder.fold(fp_hash, dimensions)
        elif return_strings:
            return atom_env_pairs
        return encoder.from_string_array(atom_env_pairs)

    except ValueError:
        # print(MolToSmiles(Mol))
        return np.full(shape=dimensions, fill_value=-1)
