from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class PatternFingerprint(BaseFingerprintTransformer):
    """
    Pattern fingerprint.

    This fingerprint is an RDKit original [1]_. This is a hashed fingerprint,
    where fragments are created from very generic SMARTS patterns, taking into
    consideration only atom and bond types (including bond aromaticity).

    For each pattern, its occurrences are detected and hashed, including
    atom and bond types involved. The fact that particular pattern matched
    the molecule at all is also stored by hashing the pattern ID and size.

    Patterns are (last four are too complex to describe easily):

    - [*] (single atom)
    - [*]~[*] (directly bonded atoms)
    - [*]~[*]~[*] (three consecutive atoms)
    - [R]~1~[R]~[R]~1 (3-atom ring)
    - [R]~1[R]~[R]~[R]~1 (4-atom ring)
    - [R]~1~[R]~[R]~[R]~[R]~1 (5-atom ring)
    - [R]~1~[R]~[R]~[R]~[R]~[R]~1 (6-atom ring)
    - [*]~[*](~[*])~[*] (atom with 3 neighbors)
    - [*]~[*]~[*](~[*])~[*]
    - [R](@[R])(@[R])~[R]~[R](@[R])(@[R])
    - [*]~[R](@[R])@[R](@[R])~[*]
    - [*]~[R](@[R])@[R]@[R](@[R])~[*]

    Parameters
    ----------
    fp_size : int, default=2048
        Size of output vectors, i.e. number of bits for each fingerprint. Must be
        positive.

    tautomers : bool, default=False
        Whether to include tautomeric bonds as a separate bond type when checking
        pattern matches.

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

    References
    ----------
    .. [1] `Gregory Landrum
        "Fingerprints in the RDKit"
        UGM 2012
        <https://www.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf>`_

    Examples
    --------
    >>> from skfp.fingerprints import PatternFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = PatternFingerprint()
    >>> fp
    PatternFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "tautomers": ["boolean"],
    }

    def __init__(
        self,
        fp_size: int = 2048,
        tautomers: bool = False,
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
        self.tautomers = tautomers

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute Pattern fingerprints.

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
        from rdkit.Chem.rdmolops import PatternFingerprint as RDKitPatternFingerprint

        X = ensure_mols(X)
        X = [
            RDKitPatternFingerprint(
                mol, fpSize=self.fp_size, tautomerFingerprints=self.tautomers
            )
            for mol in X
        ]

        if self.sparse:
            return csr_array(X, dtype=np.uint8)
        else:
            return np.array(X, dtype=np.uint8)
