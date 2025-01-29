from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class PhysiochemicalPropertiesFingerprint(BaseFingerprintTransformer):
    """
    Physiochemical properties fingerprint.

    The implementation uses RDKit. Variants of this fingerprint [1]_ are
    binding property pairs (bp) and binding property torsions (bt), based on
    Atom Pairs and Topological Torsion fingerprint, respectively.

    The difference is in atom types (invariants) used, which here correspond
    to physiochemical features of an atom. They are therefore more general,
    but relate to functional role of atoms. Atom types are:
    - cation
    - anion
    - neutral hydrogen bond donor
    - neutral hydrogen bond acceptor
    - polar atom (both donor and acceptor, e.g. hydroxy oxygen)
    - hydrophobic
    - other

    Parameters
    ----------
    fp_size : int, default=2048
        Size of output vectors, i.e. number of bits for each fingerprint. Must be
        positive.

    variant: {"BP", "BT"}, default="BP"
        Which variant to use: binding property pairs `"BP"`, or binding property
        torsions `"BT"`.

    count : bool, default=False
        Whether to return binary (bit) features, or the count-based variant.

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
    .. [1] `Simon K. Kearsley et al.
        "Chemical Similarity Using Physiochemical Property Descriptors"
        J. Chem. Inf. Comput. Sci. 1996, 36, 1, 118-127
        <https://pubs.acs.org/doi/10.1021/ci950274j>`_

    Examples
    --------
    >>> from skfp.fingerprints import PhysiochemicalPropertiesFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = PhysiochemicalPropertiesFingerprint()
    >>> fp
    PhysiochemicalPropertiesFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "variant": [StrOptions({"BP", "BT"})],
    }

    def __init__(
        self,
        fp_size: int = 2048,
        variant: str = "BP",
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
        self.variant = variant

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute Physicochemical Properties fingerprints.

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
        from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint

        X = ensure_mols(X)

        if self.variant == "BP":
            X = [GetBPFingerprint(mol) for mol in X]
        else:  # "BT" variant
            X = [GetBTFingerprint(mol) for mol in X]

        X = self._hash_fingerprint_bits(
            X, fp_size=self.fp_size, count=self.count, sparse=self.sparse
        )

        return X
