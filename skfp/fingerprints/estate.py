from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class EStateFingerprint(BaseFingerprintTransformer):
    """
    Electrotopological State (EState) fingerprint.

    The implementation uses RDKit. This is a descriptor-based fingerprint, where
    bits measure contributions of different atom types, based on their electronic
    and topological properties. Based on ``variant`` argument, either counts of
    different atom types, or sums of their properties are computed.

    The idea is to represent the intrinsic electronic state of the atom, in the context
    of the particular molecule, i.e. taking into consideration the electronic influence
    of all other atoms and the molecule topology (structure).

    79 atom types are used, as defined in the original paper [1]_. For practical
    implementation, they are formulated as SMARTS patterns, selecting individual
    atoms of particular type [2]_. Generally, they take into consideration:

    - atom element
    - valence state (including aromaticity)
    - number of bonded hydrogens
    - in some cases, the identity of other bonded atoms

    Parameters
    ----------
    variant : {"bit", "count", "sum"}, default="sum"
        Fingerprint variant. Default "sum" results in the sum of EState indices
        for each atom type. "count" results in integer vector with counts of atoms
        of particular types, and "bit" simply denotes existence of given atom types.

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
    .. [1] `Lowell H. Hall and Lemont B. Kier
        "Electrotopological State Indices for Atom Types: A Novel Combination of Electronic,
        Topological, and Valence State Information"
        J. Chem. Inf. Comput. Sci. 1995, 35, 6, 1039-1045
        <https://pubs.acs.org/doi/10.1021/ci00028a014>`_

    .. [2] `Gregory Landrum and Rational Discovery LLC
        RDKit - EState Atom Types
        <https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/EState/AtomTypes.py>`_

    Examples
    --------
    >>> from skfp.fingerprints import EStateFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = EStateFingerprint()
    >>> fp
    EStateFingerprint()

    >>> fp.transform(smiles)  # doctest: +ELLIPSIS
    array([[0.        , 0.        , ..., 0.        , 0.        ],
           [0.        , 0.        , ..., 0.        , 0.        ],
           [0.        , 0.        , ..., 0.        , 0.        ],
           [0.        , 0.        , ..., 0.        , 0.        ]])

    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "variant": [StrOptions({"bit", "count", "sum"})],
    }

    def __init__(
        self,
        variant: str = "sum",
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            n_features_out=79,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.variant = variant

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        """
        Get fingerprint output feature names. They correspond to SMARTS patterns
        defining atom types. See the original paper [1]_ for details.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            EState feature names.
        """
        from rdkit.Chem.EState.AtomTypes import _rawD

        feature_names = [smarts for name, smarts in _rawD]
        return np.asarray(feature_names, dtype=object)

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute EState fingerprints.

        Parameters
        ----------
        X : {sequence of str or Mol}
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Whether to copy input data.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 79)
            Transformed data.
        """
        return super().transform(X, copy=copy)

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.EState.Fingerprinter import FingerprintMol

        X = ensure_mols(X)

        X = np.array([FingerprintMol(mol) for mol in X])
        if self.variant == "bit":
            X = (X[:, 0] > 0).astype(np.uint8)
        elif self.variant == "count":
            X = (X[:, 0]).astype(np.uint32)
        else:  # "sum" variant
            X = X[:, 1]

        return csr_array(X) if self.sparse else np.array(X)
