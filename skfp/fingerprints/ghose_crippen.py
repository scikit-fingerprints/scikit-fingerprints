from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseSubstructureFingerprint


class GhoseCrippenFingerprint(BaseSubstructureFingerprint):
    """
    Ghose-Crippen fingerprint.

    A substructure fingerprint based on 110 atom types proposed by Ghose and
    Crippen [1]_ [2]_. They are defined for carbon, hydrogen, oxygen, nitrogen, sulfur,
    and halogens, and originally applied for predicting molar refractivities and logP.

    RDKit SMARTS patterns definitions are used [3]_.

    Parameters
    ----------
    count : bool, default=False
        Whether to return binary (bit) features, or their counts.

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

    verbose : int or dict, default=0
        Controls the verbosity when computing fingerprints.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    Attributes
    ----------
    n_features_out : int = 110
        Number of output features, size of fingerprints.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    References
    ----------
    .. [1] `Arup K. Ghose and Gordon M. Crippen
        "Atomic Physicochemical Parameters for Three-Dimensional Structure-Directed
        Quantitative Structure-Activity Relationships I. Partition Coefficients as a Measure of Hydrophobicity"
        Journal of Computational Chemistry 7.4 (1986): 565-577.
        <https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.540070419>`_

    .. [2] `Arup K. Ghose and Gordon M. Crippen
        "Atomic physicochemical parameters for three-dimensional-structure-directed
        quantitative structure-activity relationships. 2. Modeling dispersive and hydrophobic interactions"
        J. Chem. Inf. Comput. Sci. 1987, 27, 1, 21–35
        <https://pubs.acs.org/doi/10.1021/ci00053a005>`_

    .. [3] `<https://github.com/rdkit/rdkit/blob/5d034e37331c2604bf3e247b94be35b519e62216/Data/Crippen.txt>`_

    Examples
    --------
    >>> from skfp.fingerprints import GhoseCrippenFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = GhoseCrippenFingerprint()
    >>> fp
    GhoseCrippenFingerprint()

    >>> fp.transform(smiles)  # doctest: +ELLIPSIS
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 1, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 1, 0, ..., 0, 0, 0]],
          dtype=uint8)
    """

    def __init__(
        self,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        patterns = self._load_patterns()
        self._feature_names = patterns
        super().__init__(
            patterns=patterns,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Get fingerprint output feature names. They are raw SMARTS patterns
        used as feature definitions.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Names of the Ghose-Crippen feature names.
        """
        return np.asarray(self._feature_names, dtype=object)

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute Ghose-Crippen substructure fingerprints.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 307)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _load_patterns(self) -> list[str]:
        # since Ghose-Crippen file is licensed under RDKit BSD, we keep it separately
        patterns = []

        filepath = Path(__file__).parent / "data" / "Crippen.txt"
        with open(filepath) as file:
            for line in file:
                if line.startswith("#") or line.isspace():
                    continue
                smarts = line.split()[1]
                patterns.append(smarts.strip())

        return patterns
