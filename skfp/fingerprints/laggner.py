from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseSubstructureFingerprint
from skfp.utils import ensure_mols


class LaggnerFingerprint(BaseSubstructureFingerprint):
    """
    Substructure fingerprint with definitions by Christian Laggner.

    A substructure fingerprint based on SMARTS patterns for functional group
    classification, proposed by Christian Laggner [1]_. It is also known as
    SubstructureFingerprint in Chemistry Development Kit (CDK) [2]_.
    It tests for presence of 307 predefined substructures, designed for functional
    groups of organic compounds, for use in similarity searching.

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
    n_features_out : int = 307
        Number of output features, size of fingerprints.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    References
    ----------
    .. [1] `Christian Laggner
        "SMARTS Patterns for Functional Group Classification"
        OpenBabel
        <https://raw.githubusercontent.com/openbabel/openbabel/master/data/SMARTS_InteLigand.txt>`_

    .. [2] `egonw
        "SubstructureFingerprinter"
        Chemistry Development Kit (CDK) API reference
        <https://cdk.github.io/cdk/1.5/docs/api/org/openscience/cdk/fingerprint/SubstructureFingerprinter.html>`_

    Examples
    --------
    >>> from skfp.fingerprints import LaggnerFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = LaggnerFingerprint()
    >>> fp
    LaggnerFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 0],
           [1, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [1, 0, 0, ..., 1, 0, 1]], dtype=uint8)
    """

    def __init__(
        self,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        feature_names, patterns = self._load_patterns()
        self._feature_names = np.asarray(feature_names, dtype=object)
        super().__init__(
            patterns=patterns,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        """
        Get fingerprint output feature names. They correspond to substructure
        names defined by Christian Laggner, intended to capture with SMARTS patterns
        used by this fingerprint.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Laggner feature names.
        """
        return self._feature_names

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute Laggner fingerprints.

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

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem import GetMolFrags

        X = ensure_mols(X)
        fps = super()._calculate_fingerprint(X)

        # temporarily convert to dictionary-of-keys (DOK) format to set bits
        # for set  cation and salt features (setting bits on CSR array is slow)
        if self.sparse:
            fps = fps.todok()

        for idx, mol in enumerate(X):
            # salt = at least two components, has anion and cation
            # this can only be 0 or 1
            multi_component = len(GetMolFrags(mol)) > 1
            anion = fps[idx, 296]
            cation = fps[idx, 297]
            salt = multi_component & anion & cation
            fps[idx, 298] = salt

        return csr_array(fps) if self.sparse else fps

    def _load_patterns(self) -> tuple[list[str], list[str]]:
        # since Laggner file is licensed under LGPL, we keep it separately
        feature_names = []
        patterns = []

        filepath = Path(__file__).parent / "data" / "SMARTS_InteLigand.txt"
        with open(filepath) as file:
            for line in file:
                if line.startswith("#") or line.isspace():
                    continue
                elif line.startswith("Urea:"):  # this line has no space after colon
                    name = "Urea"
                    smarts = line.removeprefix("Urea:")
                else:
                    name, smarts = line.split()
                    name = name.removesuffix(":")

                feature_names.append(name.strip())
                patterns.append(smarts.strip())

        # RDKit does not support multi-component SMARTS (with a dot), so we can't match
        # the salts pattern "([-1,-2,-3,-4,-5,-6,-7]).([+1,+2,+3,+4,+5,+6,+7])"
        # (index 298); we compute it manually in .transform(), and here temporarily
        # replace it with an empty pattern
        patterns[298] = ""

        return feature_names, patterns
