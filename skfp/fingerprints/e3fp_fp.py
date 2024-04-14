import logging
from collections.abc import Sequence
from numbers import Integral, Real
from typing import Optional, Union

import numpy as np
import scipy.sparse
from e3fp.conformer.generator import ConformerGenerator
from e3fp.pipeline import fprints_from_mol
from rdkit import RDLogger
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils import Interval
from sklearn.utils._param_validation import InvalidParameterError, StrOptions

from skfp.validators import ensure_smiles

from .base import FingerprintTransformer

"""
Note: this file cannot have the "e3fp.py" name due to conflict with E3FP library.
"""


class E3FPFingerprint(FingerprintTransformer):
    """
    E3FP (Extended 3-Dimensional FingerPrint) fingerprint.

    The implementation uses `e3fp` library. This is a hashed fingerprint, where
    fragments are computed based on "shells", i.e. spherical areas around each
    atom in the 3D conformation of a molecule. The initial vector is quite large,
    and is then folded to the `fp_size` length.

    Shells are created around each atom with increasing radius, multiplied each
    time by `radius_multiplier`, until `level` iterations are reached or when there
    is no change. Shell of each radius is hashed.

    Each shells get an identifier based on atom types in their radius, which is then
    hashed. Atom types (invariants) by default are based on Daylight invariants:

    - number of heavy neighbors
    - valence (excluding hydrogen neighbors)
    - atomic number
    - atomic mass
    - formal charge
    - number of bound hydrogens
    - whether it is a part of a ring

    Note that this class generates multiple conformers, and by default selects
    the single, most stable one (with the lowest energy) for fingerprint calculation.
    Currently, it is not possible to pass precomputed conformers.

    Parameters
    ----------
    fp_size : int, default=1024
        Size of output vectors, i.e. number of bits for each fingerprint. Must be
        positive.

    n_bits_before_folding : int, default=4096
        Size of fingerprint vector after initial hashing. It is then folded to
        `fp_size`. Must be positive, and larger or equal to `fp_size`.

    level : int, default=None
        Maximal number of iterations with increasing shell radius. None means that
        it stops only when the is no change from the last iteration.

    radius_multiplier : float, default=1.718
        How much to multiply the radius by to get the next, larger shell. Must be
        greater than 1.

    rdkit_invariants : bool, default=False
        Whether to use RDKit ECFP invariants instead of Daylight ones.

    num_conf_generated : int, default=3
        Number of conformers to generate per molecule. Note that due to energy
        minimization and filtering the actual number of conformers used can be
        smaller. Must be greater or equal to 1.

    num_conf_used : int, default=1
        Number of conformers to use for fingerprint calculation per molecule.
        If more than one is used, the resulting fingerprint is aggregated for
        molecule as determined by `aggregation_type`. Must be less or equal to
        `num_conf_generated`.

    pool_multiplier : float, default=1.0
        Factor to multiply by max_conformers to generate the initial conformer pool.
        Conformers are first generated, and then filtered after energy minimization.
        Increasing the size of the pool increases the chance of identifying more
        unique conformers. Must be greater or equal to 1.

    rmsd_cutoff : float, default=0.5
        RMSD cutoff for pruning conformers. If None, no pruning is performed.

    max_energy_diff : float, default=None
        If set, conformers with energies this amount above the minimum energy
        conformer are filtered out. Must be nonnegative.

    force_field : {"uff", "mmff94", "mmff94s"}, default="uff"
        Force field optimization algorithms to use on generated conformers.

    aggregation_type : {"min_energy}", default="min_energy"
        How to aggregate fingerprints calculated from different conformers of
        a single molecule. Currently, only "min_energy" option is supported, which
        selects the fingerprint from the lowest energy (the most stable) conformer.

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

    random_state : int, RandomState instance or None, default=0
        Controls the randomness of conformer generation.

    Attributes
    ----------
    n_features_out : int
        Number of output features, size of fingerprints. Equal to `fp_size`.

    See Also
    --------
    :class:`ECFPFingerprint` : Related 2D fingerprint. E3FP was designed to extend it
        to 3D features.

    References
    ----------
    .. [1] `Axen, Seth D., et al.
        "A simple representation of three-dimensional molecular structure"
        J. Med. Chem. 2017, 60, 17, 7393–7409
        <https://pubs.acs.org/doi/abs/10.1021/acs.jmedchem.7b00696>`_

    Examples
    --------
    >>> from skfp.fingerprints import E3FPFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = E3FPFingerprint()
    >>> fp
    E3FPFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    """

    _parameter_constraints: dict = {
        **FingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "n_bits_before_folding": [Interval(Integral, 1, None, closed="left")],
        "level": [None, Interval(Integral, 1, None, closed="left")],
        "radius_multiplier": [Interval(Real, 1.0, None, closed="neither")],
        "rdkit_invariants": ["boolean"],
        "num_conf_generated": [Interval(Integral, 1, None, closed="left")],
        "num_conf_used": [Interval(Integral, 1, None, closed="left")],
        "pool_multiplier": [Interval(Real, 1.0, None, closed="left")],
        "rmsd_cutoff": [Interval(Real, 0.0, None, closed="left"), None],
        "max_energy_diff": [Interval(Real, 0.0, None, closed="left"), None],
        "force_field": [StrOptions({"uff", "mmff94", "mmff94s"})],
        "aggregation_type": [StrOptions({"min_energy"})],
    }

    def __init__(
        self,
        fp_size: int = 1024,
        n_bits_before_folding: int = 4096,
        level: Optional[int] = None,
        radius_multiplier: float = 1.718,
        rdkit_invariants: bool = False,
        num_conf_generated: int = 3,
        num_conf_used: int = 1,
        pool_multiplier: float = 1.0,
        rmsd_cutoff: Optional[float] = 0.5,
        max_energy_diff: Optional[float] = None,
        force_field: str = "uff",
        aggregation_type: str = "min_energy",
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
        random_state: int = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
        )
        self.fp_size = fp_size
        self.n_bits_before_folding = n_bits_before_folding
        self.level = level
        self.radius_multiplier = radius_multiplier
        self.rdkit_invariants = rdkit_invariants
        self.num_conf_generated = num_conf_generated
        self.num_conf_used = num_conf_used
        self.pool_multiplier = pool_multiplier
        self.rmsd_cutoff = rmsd_cutoff
        self.max_energy_diff = max_energy_diff
        self.force_field = force_field
        self.aggregation_type = aggregation_type

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.n_bits_before_folding < self.fp_size:
            raise InvalidParameterError(
                f"n_bits_before_folding must be greater of equal to fp_size, got:"
                f"n_bits_before_folding={self.n_bits_before_folding}, "
                f"fp_size={self.fp_size}"
            )
        if self.num_conf_generated < self.num_conf_used:
            raise InvalidParameterError(
                f"num_conf_generated must be greater of equal to num_conf_used, got:"
                f"num_conf_generated={self.num_conf_generated}, "
                f"num_conf_used={self.num_conf_used}"
            )

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """Compute E3FP fingerprints.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.fp_size)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(self, X: Sequence[str]) -> Union[np.ndarray, csr_array]:
        X = ensure_smiles(X)
        X = [self._calculate_single_mol_fingerprint(smi) for smi in X]
        return scipy.sparse.vstack(X) if self.sparse else np.array(X)

    def _calculate_single_mol_fingerprint(
        self, smiles: str
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem import MolFromSmiles
        from rdkit.Chem.PropertyMol import PropertyMol

        # e3fp library has a bug with passing floating point number
        # as a number of conformers, where RDKit requires an integer
        # we fix this by explicitly passing the target number of conformer
        num_conf = round(self.num_conf_generated * self.pool_multiplier)

        conf_gen = ConformerGenerator(
            first=self.num_conf_used,
            num_conf=num_conf,
            pool_multiplier=1,
            rmsd_cutoff=self.rmsd_cutoff,
            max_energy_diff=self.max_energy_diff,
            forcefield=self.force_field,
            get_values=True,
            seed=self.random_state,
        )

        mol = MolFromSmiles(smiles)
        mol.SetProp("_Name", smiles)
        mol = PropertyMol(mol)
        mol.SetProp("_SMILES", smiles)

        # Generating conformers
        # TODO: for some molecules conformers are not properly generated - returns an empty list and throws RuntimeError
        try:
            # suppress flood of logs
            if not self.verbose:
                logging.disable(logging.INFO)
                RDLogger.DisableLog("rdApp.*")

            mol, values = conf_gen.generate_conformers(mol)
            fps = fprints_from_mol(
                mol,
                fprint_params={
                    "bits": self.n_bits_before_folding,
                    "level": self.level,
                    "radius_multiplier": self.radius_multiplier,
                    "rdkit_invariants": self.rdkit_invariants,
                    "counts": self.count,
                },
            )
        finally:
            RDLogger.EnableLog("rdApp.*")
            logging.disable(logging.NOTSET)

        # TODO: add other aggregation types
        # "min_energy" aggregation
        energies = values[2]
        fp = fps[np.argmin(energies)]

        fp = fp.fold(self.fp_size)
        dtype = np.uint32 if self.count else np.uint8
        return fp.to_vector(sparse=self.sparse, dtype=dtype)
