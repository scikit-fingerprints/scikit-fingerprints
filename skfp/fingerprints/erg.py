from collections.abc import Sequence
from numbers import Integral, Real
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, InvalidParameterError, StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class ERGFingerprint(BaseFingerprintTransformer):
    """
    Extended-Reduced Graph (ERG) fingerprint.

    The implementation uses RDKit. This fingerprint is descriptor-based, but has
    variable length, depending on distance parameters. Originally described in [1]_.

    This fingerprint can be seen as a hybrid of reduced graphs [2]_ and binding
    property (BP) pairs [3]_. It uses fuzzy incrementation instead of counts.

    First, input molecule is converted into a "reduced graph", condensing
    pharmacophorically relevant information. On that graph, property points (PP) are
    identified. From those nodes, triplets (PP 1, topological distance, PP 2) are
    formed, using the shortest path distances on the reduced graph. They are not
    hashed, in contrast to Atom Pair approach, but rather all such triplets with
    distances between ``min_distance`` and ``max_distance`` are considered. For each
    triplet occurrence, its bit is incremented by 1 (resulting in triplet counts),
    and additionally, the fields for closest distances, i.e. ``dist-1`` and ``dist+1``,
    are also incremented by ``fuzz_increment`` (hence the fuzziness).

    Six features are used for both computing the reduced graph and identifying
    property points, using SMARTS patterns definitions from [4]_:
    - hydrogen bond donors
    - hydrogen bond acceptors
    - aromatic ring systems
    - hydrophobic (aliphatic) ring systems
    - positive formal atom charges
    - negative formal atom charges

    Because we always use 6 features, the resulting fingerprints have length
    ``21 * (max_distance - min_distance + 1)``.

    Note that RDKit does not implement two features from the original paper,
    flip-flop flags (for donors and acceptors), and collapsing highly fused rings.

    Parameters
    ----------
    fuzz_increment : float, default=0.3
        How much to increment triplets occurrences and their closest neighboring
        fields by when they are detected. Controls how much weight is put on the
        similarity of pharmacophoric patterns. Default value of 0.3 was optimized
        for scaffold-hopping applications, but lower values around 0.1-0.2 can be
        considered for more "crisp" similarity.

    min_path : int, default=1
        Minimal shortest path length to consider for calculating triplets on the
        reduced graph.

    max_path : int, default=15
        Maximal shortest path length to consider for calculating triplets on the
        reduced graph.

    variant : {"fuzzy", "bit", "count"}, default="fuzzy"
        Fingerprint variant. Default "fuzzy" follows the original paper and results
        in floating point results. "bit" and "count" result in integer vectors for
        "crisp" calculation on the original graph, and they use zero fuzziness,
        ignoring the ``fuzz_incremement`` parameter.

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
        Number of output features, size of fingerprints. Equal to
        ``21 * (max_path - min_path)``, 315 for default parameters.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    See Also
    --------
    :class:`TopologicalTorsionFingerprint` : Related fingerprint, but uses 4-atom paths.

    References
    ----------
    .. [1] `Nikolaus Stiefl, Ian A. Watson, Knut Baumann, and Andrea Zaliani
        "ErG: 2D Pharmacophore Descriptions for Scaffold Hopping"
        J. Chem. Inf. Model. 2006, 46, 1, 208-220
        <https://pubs.acs.org/doi/abs/10.1021/ci050457y>`_

    .. [2] `Valerie J. Gillet, Peter Willett, and John Bradshaw
        "Similarity Searching Using Reduced Graphs"
        J. Chem. Inf. Comput. Sci. 2003, 43, 2, 338-345
        <https://pubs.acs.org/doi/abs/10.1021/ci025592e>`_

    .. [3] `Simon K. Kearsley, Susan Sallamack, Eugene M. Fluder, Joseph D. Andose,
        Ralph T. Mosley, and Robert P. Sheridan
        "Chemical Similarity Using Physiochemical Property Descriptors"
        J. Chem. Inf. Comput. Sci. 1996, 36, 1, 118-127
        <https://pubs.acs.org/doi/abs/10.1021/ci950274j>`_

    .. [4] `Alberto Gobbi and Dieter Poppinger
        "Genetic optimization of combinatorial libraries"
        Biotechnol Bioeng. 1998 Winter;61(1):47-54.
        <https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-0290(199824)61:1%3C47::AID-BIT9%3E3.0.CO%3B2-Z>`_

    Examples
    --------
    >>> from skfp.fingerprints import ERGFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = ERGFingerprint()
    >>> fp
    ERGFingerprint()

    >>> fp.transform(smiles)
    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fuzz_increment": [Interval(Real, 0.0, None, closed="left")],
        "min_path": [Interval(Integral, 1, None, closed="left")],
        "max_path": [Interval(Integral, 1, None, closed="left")],
        "variant": [StrOptions({"fuzzy", "bit", "count"})],
    }

    def __init__(
        self,
        fuzz_increment: float = 0.3,
        min_path: int = 1,
        max_path: int = 15,
        variant: str = "fuzzy",
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            n_features_out=int(max_path * 21),
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.fuzz_increment = fuzz_increment
        self.min_path = min_path
        self.max_path = max_path
        self.variant = variant

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.max_path <= self.min_path:
            raise InvalidParameterError(
                f"The max_path parameter of {self.__class__.__name__} must be "
                f"greater than min_path, got: "
                f"min_path={self.min_path}, max_path={self.max_path}"
            )

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute ERG fingerprints.

        Parameters
        ----------
        X : {sequence of str or Mol}
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Whether to copy input data.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.n_features_out)
            Transformed data.
        """
        return super().transform(X, copy=copy)

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

        X = ensure_mols(X)

        fuzz = self.fuzz_increment if self.variant == "fuzzy" else 0

        X = np.array(
            [
                GetErGFingerprint(
                    mol,
                    fuzzIncrement=fuzz,
                    minPath=self.min_path,
                    maxPath=self.max_path,
                )
                for mol in X
            ]
        )

        if self.variant == "bit":
            X = X > 0
            dtype = np.uint8
        elif self.variant == "count":
            X = np.round(X)
            dtype = np.uint32  # type: ignore
        else:  # "fuzzy"
            dtype = float  # type: ignore

        if self.sparse:
            return csr_array(X, dtype=dtype)
        else:
            return np.array(X, dtype=dtype)
