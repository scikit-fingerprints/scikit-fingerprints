from collections.abc import Sequence
from numbers import Real
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols, ensure_smiles, no_rdkit_logs


class RDKit2DDescriptorsFingerprint(BaseFingerprintTransformer):
    """
    RDKit 2D descriptors fingerprint.

    The implementation uses descriptastorus [1]_ and RDKit. This fingerprint consists
    of 200 2D descriptors available in RDKit (almost all). List of all features is
    available in descriptastorus code and in the supplementary material of the original
    paper[2]_.

    Normalized variant uses cumulative distribution function (CDF) normalization, as
    proposed in [2]_. Distributions for normalization have been determined using a
    large collection of molecules from ChEMBL [3]_.

    Typical correct values should be small, but it often results in NaN or infinity
    for some descriptors. Value clipping with ``clip_val`` parameter, feature selection,
    and/or imputation should be used.

    Parameters
    ----------
    normalized : bool, default=False
        Whether to return CDF-normalized descriptor values.

    clip_val : float or None, default=2147483647
        Value to clip results at, both positive and negative ones.The default value is
        the maximal value of 32-bit integer, but should often be set lower, depending
        on the application. ``None`` means that no clipping is applied.

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
    n_features_out : int = 200
        Number of output features, size of fingerprints.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    References
    ----------
    .. [1] Descriptastorus
        https://github.com/bp-kelley/descriptastorus

    .. [2] `Kevin Yang et al.
        "Analyzing Learned Molecular Representations for Property Prediction"
        Journal of Chemical Information and Modeling 59.8 (2019): 3370-3388
        <https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237>`_

    .. [3] Descriptastorus normalized descriptors discussion
        https://github.com/bp-kelley/descriptastorus/issues/3

    Examples
    --------
    >>> from skfp.fingerprints import RDKit2DDescriptorsFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = RDKit2DDescriptorsFingerprint()
    >>> fp
    RDKit2DDescriptorsFingerprint()

    >>> fp.transform(smiles)  # doctest: +SKIP
    array([[ 1.00000000e+00,  0.00000000e+00,  ...  0.00000000e+00,  3.27747673e-01]
           [ 1.00000000e+00,  1.00000000e+00,  ...  0.00000000e+00,  3.72785568e-01]
           [ 1.00000000e+00,  3.00000000e+00,  ...  0.00000000e+00,  3.44374359e-01]
           [ 1.00000000e+00,  2.18749619e+00,  ...  0.00000000e+00,  3.55007619e-01]], dtype=float32)
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "normalized": ["boolean"],
        "clip_val": [None, Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        normalized: bool = False,
        clip_val: float = 2147483647,  # max int32 value
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            n_features_out=200,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.normalized = normalized
        self.clip_val = clip_val

    def get_feature_names_out(self, input_features=None):  # noqa: ARG002
        """
        Get fingerprint output feature names. They correspond to RDKit function
        names for computing descriptors.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            RDKit 2D descriptor names.
        """
        from descriptastorus.descriptors.rdDescriptors import RDKit2D
        from descriptastorus.descriptors.rdNormalizedDescriptors import (
            RDKit2DNormalized,
        )

        gen = RDKit2DNormalized() if self.normalized else RDKit2D()
        feature_names = [name for name, obj in gen.columns]

        return np.asarray(feature_names, dtype=object)

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute fingerprints consisting of all RDKit 2D descriptors.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 200)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from descriptastorus.descriptors.rdDescriptors import RDKit2D
        from descriptastorus.descriptors.rdNormalizedDescriptors import (
            RDKit2DNormalized,
        )

        mols = ensure_mols(X)
        smiles = ensure_smiles(X)

        # turn off RDKit logs, since descriptastorus does not use MorganGenerator
        # and generates a lot of warnings
        with no_rdkit_logs():
            gen = RDKit2DNormalized() if self.normalized else RDKit2D()
            X = [np.array(gen.calculateMol(mol, smi)) for mol, smi in zip(mols, smiles)]

        # clip values to float32 range
        X = [np.clip(x, -self.clip_val, self.clip_val) for x in X]

        if self.sparse:
            return csr_array(X, dtype=np.float32)
        else:
            return np.array(X, dtype=np.float32)
