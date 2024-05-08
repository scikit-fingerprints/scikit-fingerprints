import warnings
from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, InvalidParameterError

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import ensure_mols, require_mols_with_conf_ids


class AtomPairFingerprint(BaseFingerprintTransformer):
    """
    Atom Pair fingerprint.

    The implementation uses RDKit. This is a hashed fingerprint, where
    fragments are computed based on pairs of atoms and distance between them.

    Concretely, the hashed fragment is a triplet: (atom type 1, atom type 2, distance)

    Atom type takes into consideration:

    - atomic number
    - number of pi electrons
    - degree (number of bonds)
    - optionally: chirality (based on `include_chirality` parameter)

    Distance is normally the topological distance, i.e. length of the shortest path
    in the molecular graph (number of bonds between atoms). Only pairs with distance
    between `min_distance` and `max_distance` (both inclusive) are used.

    If `use_3D` is True, then the Euclidean distance between atoms in a conformation
    is used. Note that this uses `conf_id` property of input molecules, and requires
    them to have this property set.

    Values of count version are sensitive to the molecule size, since the number of
    shortest paths scales with square of heavy atom count (HAC). This can be offset
    by setting `scale_by_hac` to True (divide counts by HAC), or integer value greater
    than 1, which divides by HAC to the given power. Setting `scale_by_hac=2` makes
    valeus independent of molecule size.

    Parameters
    ----------
    fp_size : int, default=2048
        Size of output vectors, i.e. number of bits for each fingerprint. Must be
        positive.

    min_distance : int, default = 1
        Minimal distance between atoms. Must be positive and less or equal to
        `max_distance`.

    max_distance : int, default = 30
        Maximal distance between atoms. Must be positive and greater or equal to
        `min_distance`.

    include_chirality : bool, default=False
        Whether to include chirality information when computing atom types.

    count_simulation : bool, default=True
        Whether to use count simulation for approximating feature counts.
        See [3] for details.

    use_3D : bool, default=False
        Whether to use 3D Euclidean distance matrix. If False, uses topological
        distances on molecular graph.

    count : bool, default=False
        Whether to return binary (bit) features, or their counts.

    scale_by_hac: bool or int, default=False
        Whether to scale count fingerprint by the heavy atom count (HAC) to
        obtain a proportionality to molecule size [2]. If integer value is given,
        scaling uses given power of HAC, e.g. `scale_by_hac=2` divides counts by
        squared HAC. Values are expressed as percentages in range [0, 100].

    sparse : bool, default=False
        Whether to return dense NumPy array, or sparse SciPy CSR array.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See Scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int, default=0
        Controls the verbosity when computing fingerprints.

    Attributes
    ----------
    n_features_out : int
        Number of output features, size of fingerprints. Equal to `fp_size`.

    requires_conformers : bool
        Whether the fingerprint is 3D-based and requires molecules with conformers as
        inputs, with ``conf_id`` integer property set. This depends on the ``use_3D``
        attribute, and has the same value as that parameter.

    See Also
    --------
    :class:`TopologicalTorsionFingerprint` : Related fingerprint, but uses 4-atom paths.

    References
    ----------
    .. [1] `Raymond E. Carhart, Dennis H. Smith, and R. Venkataraghavan
        "Atom pairs as molecular features in structure-activity studies: definition
        and applications"
        J. Chem. Inf. Comput. Sci. 1985, 25, 2, 64–73
        <https://pubs.acs.org/doi/10.1021/ci00046a002>`_

    .. [2] `Mahendra Awale and Jean-Louis Reymond
        "Atom Pair 2D-Fingerprints Perceive 3D-Molecular Shape and Pharmacophores for
        Very Fast Virtual Screening of ZINC and GDB-17"
        J. Chem. Inf. Model. 2014, 54, 7, 1892–1907
        <https://pubs.acs.org/doi/10.1021/ci500232g>`_

    .. [3] `Greg Landrum
        "Simulating count fingerprints"
        RDKit blog 2021
        <https://greglandrum.github.io/rdkit-blog/posts/2021-07-06-simulating-counts.html>`_

    Examples
    --------
    >>> from skfp.fingerprints import AtomPairFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = AtomPairFingerprint()
    >>> fp
    AtomPairFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "min_distance": [Interval(Integral, 1, None, closed="left")],
        "max_distance": [Interval(Integral, 1, None, closed="left")],
        "include_chirality": ["boolean"],
        "count_simulation": ["boolean"],
        "use_3D": ["boolean"],
        "scale_by_hac": ["boolean", Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        fp_size: int = 2048,
        min_distance: int = 1,
        max_distance: int = 30,
        include_chirality: bool = False,
        count_simulation: bool = True,
        use_3D: bool = False,
        scale_by_hac: Union[bool, int] = False,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            requires_conformers=use_3D,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.fp_size = fp_size
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.include_chirality = include_chirality
        self.count_simulation = count_simulation
        self.scale_by_hac = scale_by_hac
        self.use_3D = use_3D

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.max_distance < self.min_distance:
            raise InvalidParameterError(
                f"The max_distance parameter of {self.__class__.__name__} must be "
                f"greater or equal to min_distance, got: "
                f"min_distance={self.min_distance}, max_distance={self.max_distance}"
            )

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute Atom Pair fingerprints.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit Mol objects. If `use_3D`
            is True, only Mol objects with computed conformations and with
            `conf_id` property are allowed.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.fp_size)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdFingerprintGenerator import GetAtomPairGenerator

        if self.use_3D:
            X = require_mols_with_conf_ids(X)
            conf_ids = [mol.GetIntProp("conf_id") for mol in X]
        else:
            X = ensure_mols(X)
            conf_ids = [-1 for _ in X]

        gen = GetAtomPairGenerator(
            fpSize=self.fp_size,
            minDistance=self.min_distance,
            maxDistance=self.max_distance,
            includeChirality=self.include_chirality,
            use2D=not self.use_3D,
            countSimulation=self.count_simulation,
        )
        if self.count:
            fps = [
                gen.GetCountFingerprintAsNumPy(mol, confId=conf_id)
                for mol, conf_id in zip(X, conf_ids)
            ]
        else:
            fps = [
                gen.GetFingerprintAsNumPy(mol, confId=conf_id)
                for mol, conf_id in zip(X, conf_ids)
            ]

        if self.scale_by_hac:
            if self.count:
                fps = [self._scale_by_hac(fp, mol) for fp, mol in zip(fps, X)]
            else:
                warnings.warn(
                    "Scaling by HAC can only be applied to count vectors. "
                    "No HAC scaling will be applied."
                )

        return csr_array(fps) if self.sparse else np.array(fps)

    def _scale_by_hac(self, fingerprint: np.ndarray, mol: Mol) -> np.ndarray:
        scale_factor = mol.GetNumHeavyAtoms() ** self.scale_by_hac
        return (100 * fingerprint) / scale_factor
