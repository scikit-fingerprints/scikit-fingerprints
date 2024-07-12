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

        - acyclic nodes: single, di-, tri- and tetravalent
        - cyclic nodes: di, tri- and tetravalent
        - rings with size 3, 4, ..., 9, 10 or larger
        - nodes and bonds shared by 2 or more rings

    RDKit implementation is used, and may differ slightly from the original one in
    terms of polarity due to different donor and acceptor definitions.

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
        See Scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int, default=0
        Controls the verbosity when computing fingerprints.

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
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=42,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute MQNs fingerprints.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit Mol objects.

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
