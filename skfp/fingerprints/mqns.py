from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class MQNsFingerprint(BaseFingerprintTransformer):
    """
    Molecular Quantum Numbers (MQNs) fingerprint.

    The implementation uses RDKit. This is a descriptor-based fingerprint, which
    uses counts of 42 simple structural features [1]_.

    Features can be divided into four categories:

    - atom counts (12 features):

        - atoms: C, F, Cl, Br, I, S, P
        - acyclic and cyclic nitrogens
        - acyclic and cyclic oxygens
        - heavy atom count (HAC)

    - bond counts (7 features):

        - acyclic and cyclic bonds: single, double, triple
        - rotatable bonds

    - polarity counts (6 features)

        - H-bond acceptor sites and atoms
        - H-bond donor sites and atoms
        - negative and positive charges (predicted at pH 7)

    - topology counts (for H-depleted graph, 17 features):

        - acyclic nodes with degree 1, 2, 3, 4
        - cyclic nodes with degree 2, 3, 4
        - rings with size 3, 4, ..., 9, 10 or larger
        - nodes and bonds shared by 2 or more rings

    RDKit implementation is used, and may differ slightly from the original one in
    terms of polarity due to different donor and acceptor definitions. Aromatic bonds
    are divided between single and double, with remainder going to single.

    Note that by default this fingerprint returns count, not bit features, in order
    to follow the original paper.

    Parameters
    ----------
    count : bool, default=True
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
    n_features_out : int = 42
        Number of output features, size of fingerprints.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    References
    ----------
    .. [1] `Kong T. Nguyen, Lorenz C. Blum, Ruud van Deursen, Jean-Louis Reymond
        "Classification of Organic Molecules by Molecular Quantum Numbers"
        ChemMedChem Volume 4, Issue 11 p. 1803-1805
        <https://chemistry-europe.onlinelibrary.wiley.com/doi/abs/10.1002/cmdc.200900317>`_

    Examples
    --------
    >>> from skfp.fingerprints import MQNsFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = MQNsFingerprint()
    >>> fp
    MQNsFingerprint()

    >>> fp.transform(smiles)  # doctest: +ELLIPSIS
    array([[0, 0, 0, ..., 0, 0, 0],
           [2, 0, 0, ..., 0, 0, 0],
           [1, 0, 0, ..., 0, 0, 0],
           [2, 0, 0, ..., 0, 0, 0]],
          dtype=uint32)
    """

    def __init__(
        self,
        count: bool = True,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            n_features_out=42,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        """
        Get fingerprint output feature names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            MQNs feature names.
        """
        feature_names = [
            "C atoms",
            "F atoms",
            "Cl atoms",
            "Br atoms",
            "I atoms",
            "S atoms",
            "P atoms",
            "N acyclic atoms",
            "N cyclic atoms",
            "O acyclic atoms",
            "O cyclic atoms",
            "heavy atom count (HAC)",
            "acyclic single bonds",
            "acyclic double bonds",
            "acyclic triple bonds",
            "cyclic single bonds",
            "cyclic double bonds",
            "cyclic triple bonds",
            "rotatable bonds",
            "acceptor sites",
            "acceptor atoms",
            "donor sites",
            "donor atoms",
            "negative charges",
            "positive charges",
            "acyclic atoms degree 1",
            "acyclic atoms degree 2",
            "cyclic atoms degree 2",
            "acyclic atoms degree 3",
            "cyclic atoms degree 3",
            "acyclic atoms degree 4",
            "cyclic atoms degree 4",
            "rings of size 3",
            "rings of size 4",
            "rings of size 5",
            "rings of size 6",
            "rings of size 7",
            "rings of size 8",
            "rings of size 9",
            "rings of size >= 10",
            "atoms in >= 2 rings",
            "bonds in >= 2 rings",
        ]
        return np.asarray(feature_names, dtype=object)

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute MQNs fingerprints.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 42)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import MQNs_

        X = ensure_mols(X)

        X = np.array([MQNs_(mol) for mol in X])
        if not self.count:
            X = X > 0

        dtype = np.uint32 if self.count else np.uint8
        return csr_array(X, dtype=dtype) if self.sparse else np.array(X, dtype=dtype)
