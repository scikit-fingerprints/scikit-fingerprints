from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class AvalonFingerprint(BaseFingerprintTransformer):
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

    References
    ----------
    .. [1] `Avalon toolkit
        <https://sourceforge.net/projects/avalontoolkit/>`_

    .. [2] `Gedeck, Peter, Bernhard Rohde, and Christian Bartels
        "QSAR − How Good Is It in Practice? Comparison of Descriptor Sets on an Unbiased
        Cross Section of Corporate Data Sets."
        J. Chem. Inf. Model. 2006, 46, 5, 1924-1936
        <https://pubs.acs.org/doi/abs/10.1021/ci050413p>`_

    Examples
    --------
    >>> from skfp.fingerprints import AvalonFingerprint
    >>> smiles = ["CCO", "CCN"]
    >>> fp = AvalonFingerprint()
    >>> fp
    AvalonFingerprint()
    >>> X = fp.transform(smiles)
    >>> X  # doctest: +ELLIPSIS
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        fp_size: int = 512,
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

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute Avalon fingerprints.

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
        from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP, GetAvalonFP

        X = ensure_mols(X)

        if self.count:
            X = [GetAvalonCountFP(mol, nBits=self.fp_size).ToList() for mol in X]
        else:
            X = [GetAvalonFP(mol, nBits=self.fp_size) for mol in X]

        dtype = np.uint32 if self.count else np.uint8

        if self.sparse:
            return csr_array(X, dtype=dtype)
        else:
            return np.array(X, dtype=dtype)
