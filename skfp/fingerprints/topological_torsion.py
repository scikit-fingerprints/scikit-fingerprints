from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class TopologicalTorsionFingerprint(BaseFingerprintTransformer):
    """
    Topological Torsion fingerprint.

    The implementation uses RDKit. This is a hashed fingerprint, where
    the hashed fragments are computed based on topological torsions [1]_.

    A topological torsion is defined as a linear sequence of consecutively bonded heavy (non-hydrogen):
    (atom 1 type)-(atom 2 type)-(atom 3 type)-(atom 4 type)

    Atom type takes into consideration:
    - the number of atom's pi bond electrons
    - the atomic type
    - the number of non-hydrogen branches attached to the atom

    This example of 4 atom path is the canonical version of topological torsion.
    The number of atoms can be adjusted (using `torsion_atom_count` parameter).

    Parameters
    ----------
    fp_size : int, default=2048
        Size of output vectors, i.e. number of bits for each fingerprint. Must be
        positive.

    include_chirality : bool, default=False
        Whether to include chirality information when computing atom types.

    torsion_atom_count : int, default=4
        The number of atoms to be included in the torsion.

    count_simulation : bool, default=True
        Whether to use count simulation for approximating feature counts [2]_.

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

    verbose : int, default=0
        Controls the verbosity when computing fingerprints.

    Attributes
    ----------
    n_features_out : int
        Number of output features, size of fingerprints. Equal to `fp_size`.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    See Also
    --------
    :class:`AtomPairFingerprint` : Related fingerprint, but uses 2 atoms and the distance between them.

    References
    ----------
    .. [1] `Ramaswamy Nilakantan, Norman Bauman, J. Scott Dixon, R. Venkataraghavan
        "Topological torsion: a new molecular descriptor for SAR applications. Comparison with other descriptors"
        J. Chem. Inf. Comput. Sci. 1987, 27, 82-85
        <https://pubs.acs.org/doi/10.1021/ci00054a008>`_

    .. [2] `Gregory Landrum
        "Simulating count fingerprints"
        RDKit blog 2021
        <https://greglandrum.github.io/rdkit-blog/posts/2021-07-06-simulating-counts.html>`_

    Examples
    --------
    >>> from skfp.fingerprints import TopologicalTorsionFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = TopologicalTorsionFingerprint()
    >>> fp
    TopologicalTorsionFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)

    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "include_chirality": ["boolean"],
        "torsion_atom_count": [Interval(Integral, 2, None, closed="left")],
        "count_simulation": ["boolean"],
    }

    def __init__(
        self,
        fp_size: int = 2048,
        include_chirality: bool = False,
        torsion_atom_count: int = 4,
        count_simulation: bool = True,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
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
        self.include_chirality = include_chirality
        self.torsion_atom_count = torsion_atom_count
        self.count_simulation = count_simulation

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute Topological Torsion fingerprints.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit Mol objects.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.fp_size)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdFingerprintGenerator import GetTopologicalTorsionGenerator

        X = ensure_mols(X)

        gen = GetTopologicalTorsionGenerator(
            includeChirality=self.include_chirality,
            torsionAtomCount=self.torsion_atom_count,
            countSimulation=self.count_simulation,
            fpSize=self.fp_size,
        )

        if self.count:
            X = [gen.GetCountFingerprintAsNumPy(mol) for mol in X]
        else:
            X = [gen.GetFingerprintAsNumPy(mol) for mol in X]

        dtype = np.uint32 if self.count else np.uint8
        if self.sparse:
            return csr_array(X, dtype=dtype)
        else:
            return np.array(X, dtype=dtype)
