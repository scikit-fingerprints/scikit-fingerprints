import itertools
import struct
from collections import defaultdict
from collections.abc import Sequence
from hashlib import sha256
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol, MolToSmiles, PathToSubmol
from rdkit.Chem.rdmolops import FindAtomEnvironmentOfRadiusN, GetDistanceMatrix
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class MAPFingerprint(BaseFingerprintTransformer):
    """
    MinHashed Atom Pair fingerprint (MAP).

    Implementation is based on the official MAP4 paper and code [1]_ [2]_. This is a
    hashed fingerprint, using the ideas from Atom Pair and SECFP fingerprints.

    It computes fragments based on pairs of atoms, using circular
    substructures around each atom represented with SMILES (like SECFP) and length
    of shortest path between them (like Atom Pair), and then hashes the resulting
    triplet using MinHash fingerprint.

    Subgraphs are created around each atom with increasing radius, starting
    with just an atom itself. It is then transformed into a canonical SMILES.
    In each iteration, it is increased by another atom (one "hop" on the graph).

    For each pair of atoms, length of the shortest path is created. Then, for
    each radius from 1 up to the given maximal radius, the triplet is created:
    (atom 1 SMILES, shortest path length, atom 2 SMILES). They are then hashed
    using MinHash algorithm.

    Parameters
    ----------
    fp_size : int, default=2048
        Size of output vectors. Depending on the ``variant`` argument, those are either
        raw hashes, bits, or counts. Must be positive.

    radius : int, default=2
        Number of iterations performed, i.e. maximum radius of resulting subgraphs.
        Another common notation uses diameter, therefore MAP4 has radius 2.

    include_duplicated_shingles : bool, default=False
        Whether to include duplicated shingles in the final fingerprint.

    count : bool, default=False
        Whether to return binary (bit) features, or their counts.

    sparse : bool, default=False
        Whether to return dense NumPy array, or sparse SciPy CSR array.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when computing fingerprints.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    Attributes
    ----------
    n_features_out : int
        Number of output features. Equal to ``fp_size``.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    See Also
    --------
    :class:`AtomPairFingerprint` : Related fingerprint, which uses atom invariants and
        folding.

    :class:`SECFPFingerprint` : Related fingerprint, which only uses SMILES of circular
        subgraph around each atom.

    References
    ----------
    .. [1] `Alice Capecchi, Daniel Probst and Jean-Louis Reymond
        "One molecular fingerprint to rule them all: drugs, biomolecules, and the metabolome"
        J Cheminform 12, 43 (2020)
        <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00445-4>`_

    .. [2] `https://github.com/reymond-group/map4`

    Examples
    --------
    >>> from skfp.fingerprints import MAPFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = MAPFingerprint()
    >>> fp
    MAPFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "radius": [Interval(Integral, 0, None, closed="left")],
        "include_duplicated_shingles": [bool],
        "count": [bool],
    }

    def __init__(
        self,
        fp_size: int = 1024,
        radius: int = 2,
        include_duplicated_shingles: bool = False,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
        random_state: Optional[int] = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
        )
        self.fp_size = fp_size
        self.radius = radius
        self.include_duplicated_shingles = include_duplicated_shingles
        self.count = count

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute MAP fingerprints.

        Parameters
        ----------
        X : {sequence of str or Mol}
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Whether to copy input data.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.fp_size)
            Transformed data.
        """
        return super().transform(X, copy=copy)

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        X = ensure_mols(X)
        X = np.stack(
            [self._calculate_single_mol_fingerprint(mol) for mol in X],
            dtype=np.uint32 if self.count else np.uint8,
        )

        return csr_array(X) if self.sparse else np.array(X)

    def _calculate_single_mol_fingerprint(self, mol: Mol) -> np.ndarray:
        atoms_envs = self._get_atom_envs(mol)
        shinglings = self._get_atom_pair_shingles(mol, atoms_envs)

        folded = np.zeros(self.fp_size, dtype=np.uint32 if self.count else np.uint8)
        for shingling in shinglings:
            hashed = struct.unpack("<I", sha256(shingling).digest()[:4])[0]
            if self.count:
                folded[hashed % self.fp_size] += 1
            else:
                folded[hashed % self.fp_size] = 1
        return folded

    @classmethod
    def _find_env(cls, mol: Mol, atom_identifier: int, radius: int) -> Optional[str]:
        # get SMILES of atom environment at given radius
        atom_identifiers_within_radius: list[int] = FindAtomEnvironmentOfRadiusN(
            mol=mol, radius=radius, rootedAtAtom=atom_identifier
        )
        atom_map: dict = {}

        sub_molecule: Mol = PathToSubmol(
            mol, atom_identifiers_within_radius, atomMap=atom_map
        )
        if atom_identifier not in atom_map:
            return None

        smiles = MolToSmiles(
            sub_molecule,
            rootedAtAtom=atom_map[atom_identifier],
            # From the original implementation, which does not use isomeric SMILES.
            isomericSmiles=False,
        )
        return smiles

    def _get_atom_envs(self, mol: Mol) -> dict[int, list[Optional[str]]]:
        """
        For each atom get its environment, i.e. radius-hop neighborhood.
        """
        atoms_env: dict[int, list[Optional[str]]] = defaultdict(list)
        for atom in mol.GetAtoms():
            atom_identifier = atom.GetIdx()
            for radius in range(1, self.radius + 1):
                atom_env = MAPFingerprint._find_env(mol, atom_identifier, radius)
                atoms_env[atom_identifier].append(atom_env)
        return atoms_env

    def _get_atom_pair_shingles(self, mol: Mol, atoms_envs: dict) -> set[bytes]:
        # get a list of atom shingles as SMILES, i.e. circular structures
        # around atom pairs and the length of shortest path
        atom_pairs: set[bytes] = set()

        distance_matrix = GetDistanceMatrix(mol)
        num_atoms = mol.GetNumAtoms()
        shingle_dict: dict[str, int] = defaultdict(int)

        for idx1, idx2 in itertools.combinations(range(num_atoms), 2):
            dist = int(distance_matrix[idx1][idx2])

            for i in range(self.radius):
                env_a: Optional[str] = atoms_envs[idx1][i]
                env_b: Optional[str] = atoms_envs[idx2][i]

                shingle = self._make_shingle(env_a, env_b, dist)

                if self.include_duplicated_shingles:
                    shingle_dict[shingle] += 1
                    shingle += f"|{shingle_dict[shingle]}"

                atom_pairs.add(shingle.encode("utf-8"))

        return atom_pairs

    @staticmethod
    def _make_shingle(env_a: Optional[str], env_b: Optional[str], distance: int) -> str:
        env_a = env_a if env_a else ""
        env_b = env_b if env_b else ""

        if len(env_a) > len(env_b):
            larger_env: str = env_a
            smaller_env: str = env_b
        else:
            larger_env = env_b
            smaller_env = env_a

        shingle = f"{smaller_env}|{distance}|{larger_env}"
        return shingle
