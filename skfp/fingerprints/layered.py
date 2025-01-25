from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, InvalidParameterError

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class LayeredFingerprint(BaseFingerprintTransformer):
    """
    Layered fingerprint.

    This fingerprint is an RDKit original [1]_. This is a hashed fingerprint,
    where fragments are created from small subgraphs on the molecular graph.

    For a given molecule, all paths between ``min_path`` and ``max_path`` (inclusive)
    are extracted. Those are any subgraphs, unless ``linear_paths_only`` is set to True.
    Note that all explicit atoms, including hydrogens if present, are used.

    Then each subgraph is hashed in "layers" (hence the fingerprint name), using
    different atom and bond invariants (features). Additionally, information about path
    length and the number of distinct atoms is included. Those hashes are combined into
    a single value and hashed into the resulting fingerprint, which is folded at the end.

    Layers are:
    - pure topology (using only subgraph "shape")
    - bond order (type), ignoring aromaticiyt (aromatic bonds are treated as single)
    - atom types (atomic numbers)
    - presence of rings (whether bond is in a ring)
    - ring sizes (size of smallest ring that bond is a part of)
    - aromaticity (whether bond is aromatic or not)

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

    linear_paths_only : bool, default=False
        Whether to use only linear paths, instead of any subgraphs.

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
    :class:`RDKitFingerprint` : Related fingerprint, but uses constant bond types
        and pseudorandom numbers to set multiple bits.

    References
    ----------
    .. [1] `Gregory Landrum
        "Fingerprints in the RDKit"
        UGM 2012
        <https://www.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf>`_

    Examples
    --------
    >>> from skfp.fingerprints import LayeredFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = LayeredFingerprint()
    >>> fp
    LayeredFingerprint()

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
        "linear_paths_only": ["boolean"],
    }

    def __init__(
        self,
        fp_size: int = 2048,
        min_path: int = 1,
        max_path: int = 7,
        linear_paths_only: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.fp_size = fp_size
        self.min_path = min_path
        self.max_path = max_path
        self.linear_paths_only = linear_paths_only

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
        Compute Layered fingerprints.

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
        from rdkit.Chem.rdmolops import LayeredFingerprint as RDKitLayeredFingerprint

        X = ensure_mols(X)
        X = [
            RDKitLayeredFingerprint(
                mol,
                fpSize=self.fp_size,
                minPath=self.min_path,
                maxPath=self.max_path,
                branchedPaths=not self.linear_paths_only,
            )
            for mol in X
        ]

        if self.sparse:
            return csr_array(X, dtype=np.uint8)
        else:
            return np.array(X, dtype=np.uint8)
