from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class ECFPFingerprint(BaseFingerprintTransformer):
    """
    Extended Connectivity FingerPrint (ECFP).

    The implementation uses RDKit. This is a hashed fingerprint [1]_, where
    fragments are computed based on circular substructures around each atom.
    Also known as Morgan fingerprint.

    Subgraphs are created around each atom with increasing radius, starting
    with just an atom itself. In each iteration, it is increased by another
    atom (one "hop" on the graph). Each subgraph during iteration is hashed,
    and the resulting hashes are folded to the ``fp_size`` length.

    Each subgraph gets an identifier based on atom types in its radius, which is
    then hashed. Atom types (invariants) by default are based on Daylight invariants:

    - number of heavy neighbors
    - valence (excluding hydrogen neighbors)
    - atomic number
    - atomic mass
    - formal charge
    - number of bound hydrogens
    - whether it is a part of a ring

    Alternatively, pharmacophoric invariants can be used, representing functional
    class of atoms, resulting in FCFP (Functional Circular FingerPrint) fingerprints.

    Parameters
    ----------
    fp_size : int, default=2048
        Size of output vectors, i.e. number of bits for each fingerprint. Must be
        positive.

    radius : int, default=2
        Number of iterations performed, i.e. maximum radius of resulting subgraphs.
        Another common notation uses diameter, therefore ECFP4 has radius 2.

    use_pharmacophoric_invariants : bool, default=False
        Whether to use pharmacophoric invariants (atom types) instead of default ones.
        They are: Donor, Acceptor, Aromatic, Halogen, Basic, Acidic. This results in
        FCFP (Functional Connectivity FingerPrint) fingerprint.

    include_chirality : bool, default=False
        Whether to include chirality information when computing atom types.

    use_bond_types : bool, default=True
        Whether to use bond types (single, double, triple, aromatic) when computing
        subgraphs.

    include_ring_membership : bool, default=True
        Whether to check if atom is part of a ring when computing atom types.

    count_simulation : bool, default=False
        Whether to use count simulation for approximating feature counts [2]_.

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
    :class:`SECFPFingerprint` : Related fingerprint, which additionally uses shortest paths
        between atoms like Atom Pair fingerprint.

    :class:`MHFPFingerprint` : Related fingerprint, which uses MinHash to perform hashing
        and can compute raw hashes, instead of folded vectors.

    References
    ----------
    .. [1] `David Rogers and Mathew Hahn
        "Extended-Connectivity Fingerprints"
        J. Chem. Inf. Model. 2010, 50, 5, 742-754
        <https://pubs.acs.org/doi/10.1021/ci100050t>`_

    .. [2] `Gregory Landrum
        "Simulating count fingerprints"
        <https://greglandrum.github.io/rdkit-blog/posts/2021-07-06-simulating-counts.html>`_

    Examples
    --------
    >>> from skfp.fingerprints import ECFPFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = ECFPFingerprint()
    >>> fp
    ECFPFingerprint()

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
        "use_pharmacophoric_invariants": ["boolean"],
        "include_chirality": ["boolean"],
        "use_bond_types": ["boolean"],
        "include_ring_membership": ["boolean"],
        "use_2D": ["boolean"],
    }

    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 2,
        use_pharmacophoric_invariants: bool = False,
        include_chirality: bool = False,
        use_bond_types: bool = True,
        include_ring_membership: bool = True,
        count_simulation: bool = False,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.fp_size = fp_size
        self.radius = radius
        self.use_pharmacophoric_invariants = use_pharmacophoric_invariants
        self.include_chirality = include_chirality
        self.use_bond_types = use_bond_types
        self.include_ring_membership = include_ring_membership
        self.count_simulation = count_simulation

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute ECFP fingerprints.

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
        from rdkit.Chem.rdFingerprintGenerator import (
            GetMorganFeatureAtomInvGen,
            GetMorganGenerator,
        )

        X = ensure_mols(X)

        if self.use_pharmacophoric_invariants:
            inv_gen = GetMorganFeatureAtomInvGen()
        else:
            inv_gen = None

        gen = GetMorganGenerator(
            fpSize=self.fp_size,
            radius=self.radius,
            atomInvariantsGenerator=inv_gen,
            includeChirality=self.include_chirality,
            useBondTypes=self.use_bond_types,
            includeRingMembership=self.include_ring_membership,
            countSimulation=self.count_simulation,
        )

        if self.count:
            X = [gen.GetCountFingerprintAsNumPy(mol) for mol in X]
        else:
            X = [gen.GetFingerprintAsNumPy(mol) for mol in X]

        return csr_array(X) if self.sparse else np.array(X)
