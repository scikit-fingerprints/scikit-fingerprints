from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols, require_mols_with_conf_ids


class PharmacophoreFingerprint(BaseFingerprintTransformer):
    """
    Pharmacophore fingerprint.

    The implementation uses RDKit. This is a hashed fingerprint, where fragments are
    computed based on N-point tuples, using pharmacophoric points.

    An N-point pharmacophoric structure encodes N pharmacophoric points and pairwise
    distances between them, e.g. 3-point pharmacophore uses 6-element tuples
    (P1 D12 P2 D23 P3 D13). P is a pharmacophoric point, atom or subgraph, of a particular
    type (see below), and Dij is a topological distance (shortest path) between points
    i and j. Distance values are limited to 8 (higher values are capped at 8).

    Pharmacophoric point types are (based on SMARTS patterns definitions from [1]_):
    - hydrophobic atom
    - hydrogen bond donor
    - hydrogen bond acceptor
    - aromatic attachment
    - aliphatic attachment
    - "unusual" atom (not H, C, N, O, F, S, Cl, Br, I)
    - basic group
    - acidic group

    Those structures can be returned as raw bits, results in 39972-element vector. By
    default, they are folded into a shorter length vector. Both 2-point and 3-point
    pharmacophores (pharmacophoric pairs and triangles) are used.

    Parameters
    ----------
    variant: {"raw_bits", "bit", "count"} = "raw_bits"
        Whether to fold the raw bits output of the fingerprint into the size defined
        by fp_size. If set to ``"count"`` the occurences will be summed.

    fp_size : int, default=2048
        Size of output vectors, i.e. number of bits for each fingerprint. Must be
        positive.

    use_3D : bool, default=False
        Whether to use 3D Euclidean distance matrix, instead of topological distance.
        Binning is used to discretize values into values 0-8.

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
        parameter, and has the same value.

    References
    ----------
    .. [1] `A Gobbi, D Poppinger
        "Genetic optimization of combinatorial libraries"
        Biotechnology and Bioengineering: Volume 61, Issue 1, Winter 1998, Pages 47-54
        <https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-0290(199824)61:1%3C47::AID-BIT9%3E3.0.CO;2-Z>`_

    Examples
    --------
    >>> from skfp.fingerprints import PharmacophoreFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = PharmacophoreFingerprint()
    >>> fp
    PharmacophoreFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)

    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "variant": [StrOptions({"raw_bits", "bit", "count"})],
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "use_3D": ["boolean"],
    }

    def __init__(
        self,
        variant: str = "raw_bits",
        fp_size: int = 2048,
        use_3D: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        n_features_out = 39972 if variant == "raw_bits" else fp_size
        super().__init__(
            n_features_out=n_features_out,
            requires_conformers=use_3D,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.variant = variant
        self.fp_size = fp_size
        self.use_3D = use_3D

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem import Get3DDistanceMatrix
        from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
        from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint

        factory = Gobbi_Pharm2D.factory

        if not self.use_3D:
            X = ensure_mols(X)
            X = [Gen2DFingerprint(mol, factory) for mol in X]
        else:
            X = require_mols_with_conf_ids(X)
            X = [
                Gen2DFingerprint(
                    mol,
                    factory,
                    dMat=Get3DDistanceMatrix(mol, confId=mol.GetIntProp("conf_id")),
                )
                for mol in X
            ]

        if self.variant in {"bit", "count"}:
            # X at this point is a list of RDKit fingerprints, but MyPy doesn't get it
            return self._hash_fingerprint_bits(
                X,  # type: ignore
                fp_size=self.fp_size,
                count=(self.variant == "count"),
                sparse=self.sparse,
            )
        else:
            return (
                csr_array(X, dtype=np.uint8)
                if self.sparse
                else np.array(X, dtype=np.uint8)
            )
