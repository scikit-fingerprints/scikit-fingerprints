from numbers import Integral
from typing import Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils import Interval

from skfp.validators import ensure_mols

from .base import FingerprintTransformer


class AvalonFingerprint(FingerprintTransformer):
    """
    Avalon fingerprint.

    The implementation uses RDKit and Avalon Toolkit [1]_. This is a hashed fingerprint, where
    fragments are computed based on atom environments. The fingerprint is based on
    multiple features, including atom types, bond types, rings and paths. The detailed description
    can be found in the original paper [2]_.

    Parameters
    ----------

    fp_size : int, default=512
        Size of output vectors, i.e. number of bits for each fingerprint. Must be
        positive.

    count : bool, default=False
        Whether to use binary or count fingerprints.

    sparse : bool, default=False
        Whether to return sparse matrix.

    n_jobs : int, default=None
        Number of parallel jobs. If -1, then the number of jobs is set to the number of CPU cores.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    n_features_out : int
        Number of output features. Equal to `fp_size`.

    References
    ----------
    .. [1] Avalon toolkit
        https://sourceforge.net/projects/avalontoolkit/
    .. [2] `Gedeck, Peter, Bernhard Rohde, and Christian Bartels
        "QSAR − How Good Is It in Practice? Comparison of Descriptor Sets on an Unbiased
        Cross Section of Corporate Data Sets."
        J. Chem. Inf. Model. 2006, 46, 5, 1924–1936
        <https://pubs.acs.org/doi/abs/10.1021/ci050413p>`_

    Examples
    --------
    >>> from skfp.fingerprints import AvalonFingerprint
    >>> smiles = ["CCO", "CCN"]
    >>> fp = AvalonFingerprint()
    >>> fp
    AvalonFingerprint()
    >>> X = fp.transform(smiles)
    >>> X
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]])
    >>> X.shape
    (2, 512)
    """

    _parameter_constraints: dict = {
        **FingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        fp_size: int = 512,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.fp_size = fp_size

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """Compute Avalon fingerprints.

        Parameters
        ----------
        X : {sequence of str or Mol}
            Sequence containing SMILES strings or RDKit molecules.

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
        from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP, GetAvalonFP

        X = ensure_mols(X)

        if self.count:
            X = [GetAvalonCountFP(x, nBits=self.fp_size).ToList() for x in X]
        else:
            X = [GetAvalonFP(x, nBits=self.fp_size) for x in X]

        return csr_array(X) if self.sparse else np.array(X)
