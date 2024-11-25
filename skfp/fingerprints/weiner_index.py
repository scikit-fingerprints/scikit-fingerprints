from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from rdkit.Chem import GetDistanceMatrix, Mol
from scipy.sparse import csr_array

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class WeinerIndexFingerprint(BaseFingerprintTransformer):
    """
    Weiner Index fingerprint.

    This implementation uses RDKit to calculate the Weiner Index for molecular graphs.
    The Weiner Index is computed as the sum of all pairwise distances in the molecular
    graph's distance matrix.

    Parameters
    ----------
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
    n_features_out : int = 1
        Number of output features, which is 1 for the Weiner Index.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    Examples
    --------
    >>> from skfp.fingerprints import WeinerIndexFingerprint
    >>> smiles = ["CCO", "CC", "O=C=O"]
    >>> fp = WeinerIndexFingerprint()
    >>> fp
    WeinerIndexFingerprint()

    >>> fp.transform(smiles)
    array([[12.],
           [ 6.],
           [ 2.]])
    """

    def __init__(
        self,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=1,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:

        X = ensure_mols(X)

        weiner_indices = []
        for mol in X:
            if mol is None:
                weiner_indices.append(0.0)
                continue

            distance_matrix = GetDistanceMatrix(mol)

            weiner_index = np.sum(distance_matrix) / 2
            weiner_indices.append(weiner_index)

        weiner_indices = np.array(weiner_indices, dtype=np.float64).reshape(-1, 1)

        return csr_array(weiner_indices) if self.sparse else weiner_indices
