from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, InvalidParameterError

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class RDKitFingerprint(BaseFingerprintTransformer):
    """
    RDKit fingerprint.

    This fingerprint is an RDKit original [1]_. This is a hashed fingerprint,
    where fragments are created from small subgraphs on the molecular graph.

    For a given molecule, all paths between ``min_path`` and ``max_path`` (inclusive)
    are extracted and hashed, based on bond invariants (see below). Those are any
    subgraphs, unless ``linear_paths_only`` is set to True. Note that all explicit
    atoms, including hydrogens if present, are used.

    Each subgraph is hashed. Based on this hash value, ``nBitsPerHash`` pseudorandom
    numbers are generated and used to set bits in the resulting fingerprint. Finally,
    it is folded to ``fp_size`` length.

    Subgraphs are identified based on bonds constituting them. Bonds invariants (types,
    features) take into consideration:

    - atomic numbers and aromaticity of bonded atoms
    - degrees of bonded atoms
    - bond type/order (single, double, triple, aromatic)

    For more details on fingerprints of this type, see Daylight documentation [2]_.

    Parameters
    ----------
    fp_size : int, default=2048
        Size of output vectors, i.e. number of bits for each fingerprint. Must be
        positive.

    min_path : int, default=1
        Minimal length of paths used, in bonds. Default value means that at least
        2-atom subgraphs are used.

    max_path : int, default=7
        Maximal length of paths used, in bonds.

    use_pharmacophoric_invariants: bool, default=False
        Whether to use pharmacophoric invariants (atom types) instead of default ones.
        They are the same as in the FCFP fingerprint: Donor, Acceptor, Aromatic,
        Halogen, Basic, Acidic.

    use_bond_order : bool, default=True
        Whether to take bond order (type) into consideration when hashing subgraphs.
        False means that only graph topology (subgraph shape) is used.

    num_bits_per_feature : int, default=2
        How many bits to set for each subgraph.

    linear_paths_only : bool, default=False
        Whether to use only linear paths, instead of any subgraphs.

    count_simulation : bool, default=True
        Whether to use count simulation for approximating feature counts [3]_.

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
        Number of output features, size of fingerprints. Equal to ``fp_size``.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    See Also
    --------
    :class:`LayeredFingerprint` : Related fingerprint, but uses different atom and bond
        types to set multiple bits.

    References
    ----------
    .. [1] `Gregory Landrum
        "Fingerprints in the RDKit"
        UGM 2012
        <https://www.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf>`_

    .. [2] `Daylight documentation
        "Fingerprints - Screening and Similarity"
        <https://www.daylight.com/dayhtml/doc/theory/theory.finger.html>`_

    .. [3] `Greg Landrum
        "Simulating count fingerprints"
        <https://greglandrum.github.io/rdkit-blog/posts/2021-07-06-simulating-counts.html>`_

    Examples
    --------
    >>> from skfp.fingerprints import RDKitFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = RDKitFingerprint()
    >>> fp
    RDKitFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "min_path": [Interval(Integral, 1, None, closed="left")],
        "max_path": [Interval(Integral, 1, None, closed="left")],
        "use_bond_order": ["boolean"],
        "num_bits_per_feature": [Interval(Integral, 1, None, closed="left")],
        "linear_paths_only": ["boolean"],
        "count_simulation": ["boolean"],
    }

    def __init__(
        self,
        fp_size: int = 2048,
        min_path: int = 1,
        max_path: int = 7,
        use_pharmacophoric_invariants: bool = False,
        use_bond_order: bool = True,
        num_bits_per_feature: int = 2,
        linear_paths_only: bool = False,
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
        self.min_path = min_path
        self.max_path = max_path
        self.use_pharmacophoric_invariants = use_pharmacophoric_invariants
        self.use_bond_order = use_bond_order
        self.num_bits_per_feature = num_bits_per_feature
        self.linear_paths_only = linear_paths_only
        self.count_simulation = count_simulation

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.max_path < self.min_path:
            raise InvalidParameterError(
                f"The max_path parameter of {self.__class__.__name__} must be "
                f"greater or equal to min_path, got: "
                f"min_path={self.min_path}, max_path={self.max_path}"
            )

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute RDKit fingerprints.

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
            GetRDKitFPGenerator,
        )

        X = ensure_mols(X)

        if self.use_pharmacophoric_invariants:
            inv_gen = GetMorganFeatureAtomInvGen()
        else:
            inv_gen = None

        gen = GetRDKitFPGenerator(
            fpSize=self.fp_size,
            minPath=self.min_path,
            maxPath=self.max_path,
            atomInvariantsGenerator=inv_gen,
            useBondOrder=self.use_bond_order,
            numBitsPerFeature=self.num_bits_per_feature,
            branchedPaths=not self.linear_paths_only,
            countSimulation=self.count_simulation,
        )

        if self.count:
            X = [gen.GetCountFingerprintAsNumPy(mol) for mol in X]
        else:
            X = [gen.GetFingerprintAsNumPy(mol) for mol in X]

        dtype = np.uint32 if self.count else np.uint8
        if self.sparse:
            return csr_array(X, dtype=dtype)
        else:
            return np.array(X, dtype=dtype)
