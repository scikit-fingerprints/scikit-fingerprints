from collections.abc import Sequence
from copy import deepcopy
from numbers import Real
from typing import Optional, Union

import numpy as np
from numpy.linalg import norm
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from scipy.stats import moment
from sklearn.utils._param_validation import Interval, StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import require_mols_with_conf_ids


class ElectroShapeFingerprint(BaseFingerprintTransformer):
    """
    ElectroShape fingerprint.

    This is a descriptor-based fingerprint, extending the USR fingerprint by
    additionally considering atomic partial charges [1]_.

    It first computes atomic partial charges, and then uses both conformational
    (spatial) structure, and this electric information, to compute reference
    points (centroids). First three are like in USR, and last two
    additionally use partial charge in distance calculation. See the original paper
    [1]_ for details. For each centroid, the distribution of distances between atoms
    and the centroid is aggregated using the first three moments (mean, standard
    deviation, cubic root of skewness). This results in 15 features.

    This is a 3D fingerprint, and requries molecules with ``conf_id`` integer property
    set. They can be generated with :class:`~skfp.preprocessing.ConformerGenerator`.
    Furthermore, only molecules with 3 or more atoms are allowed, to allow computation
    of all three moments.

    Typical correct values should be small, but problematic molecules may result in NaN
    values for some descriptors. In those cases, imputation should be used.

    Parameters
    ----------
    partial_charge_model : {"Gasteiger", "MMFF94", "formal", "precomputed"}, default="formal"
        Which model to use to compute atomic partial charges. Default ``"formal"``
        computes formal charges, and is the simplest and most error-resistantone.
        ``"precomputed"`` assumes that the inputs are RDKit PropertyMol objects
        with "charge" float property set.

    charge_scaling_factor : float, default=25.0
        Partial charges are multiplied by this factor to bring them to a value
        range comparable to distances in Angstroms.

    charge_errors : {"raise", "ignore", "zero"}, default="raise"
        How to handle errors during calculation of atomic partial charges. ``"raise"``
        immediately raises any errors. ``"NaN"`` ignores any atoms that failed the
        computation; note that if all atoms fail, the error will be raised (use
        ``errors`` parameter to control this). ``"zero"`` uses default value of 0 to
        fill all problematic charges.

    errors : {"raise", "NaN", "ignore"}, default="raise"
        How to handle errors during fingerprint calculation. ``"raise"`` immediately
        raises any errors. ``"NaN"`` returns NaN values for molecules which resulted in
        errors. ``"ignore"`` suppresses errors and does not return anything for
        molecules with errors. This potentially results in less output vectors than
        input molecules, and should be used with caution.

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
    n_features_out : int = 15
        Number of output features, size of fingerprints.

    requires_conformers : bool = True
        Value is always True, as this fingerprint is 3D based. It always requires
        molecules with conformers as inputs, with ``conf_id`` integer property set.

    See Also
    --------
    :class:`USR` : Related fingerprint, which ElectroShape expands.

    :class:`USRCAT` : Related fingerprint, which expands USR with pharmacophoric
        atom types, instead of partial charges.

    References
    ----------
    .. [1] `Armstrong, M.S., Morris, G.M., Finn, P.W. et al.
        "ElectroShape: fast molecular similarity calculations incorporating shape, chirality and electrostatics"
        J Comput Aided Mol Des 24, 789–801 (2010)
        <https://doi.org/10.1007/s10822-010-9374-0>`_

    Examples
    --------
    >>> from skfp.fingerprints import ElectroShapeFingerprint
    >>> from skfp.preprocessing import MolFromSmilesTransformer, ConformerGenerator
    >>> smiles = ["CC=O"]
    >>> fp = ElectroShapeFingerprint()
    >>> fp
    ElectroShapeFingerprint()

    >>> mol_from_smiles = MolFromSmilesTransformer()
    >>> mols = mol_from_smiles.transform(smiles)
    >>> conf_gen = ConformerGenerator()
    >>> mols = conf_gen.transform(mols)
    >>> fp.transform(mols)  # doctest: +SKIP
    array([[ 4.84903774,  5.10822298, ...        ,  5.14008906,  2.75483277 ]])
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "partial_charge_model": [
            StrOptions({"Gasteiger", "MMFF94", "formal", "precomputed"})
        ],
        "charge_scaling_factor": [Interval(Real, 0.0, None, closed="neither")],
        "charge_errors": [StrOptions({"raise", "ignore", "zero"})],
        "errors": [StrOptions({"raise", "NaN", "ignore"})],
    }

    def __init__(
        self,
        partial_charge_model: str = "formal",
        charge_scaling_factor: float = 25.0,
        charge_errors: str = "raise",
        errors: str = "raise",
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=15,
            requires_conformers=True,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.partial_charge_model = partial_charge_model
        self.charge_scaling_factor = charge_scaling_factor
        self.charge_errors = charge_errors
        self.errors = errors

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute ElectroShape fingerprints.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit Mol objects, with conformers generated and
            ``conf_id`` integer property set.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.fp_size)
            Array with fingerprints.
        """
        y = np.empty(len(X))
        X, _ = self.transform_x_y(X, y, copy=copy)
        return X

    def transform_x_y(
        self, X: Sequence[Mol], y: np.ndarray, copy: bool = False
    ) -> tuple[Union[np.ndarray, csr_array], np.ndarray]:
        """
        Compute ElectroShape fingerprints. The returned values for X and y are
        properly synchronized.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit Mol objects, with conformers generated and
            ``conf_id`` integer property set.

        y : np.ndarray of shape (n_samples,)
            Array with labels for molecules.

        copy : bool, default=False
            Copy the inputs X and y or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.fp_size)
            Array with fingerprints.

        y : np.ndarray of shape (n_samples,)
            Array with labels for molecules.
        """
        if copy:
            X = deepcopy(X)
            y = deepcopy(y)

        X = super().transform(X)

        if self.errors == "ignore":
            # errors are marked as NaN rows
            idxs_to_keep = [idx for idx, x in enumerate(X) if not np.any(np.isnan(x))]
            X = X[idxs_to_keep]
            y = y[idxs_to_keep]

        return X, y

    def _calculate_fingerprint(self, X: Sequence[Mol]) -> Union[np.ndarray, csr_array]:
        mols = require_mols_with_conf_ids(X)

        if self.errors == "raise":
            fps = [self._get_fp(mol) for mol in mols]
        else:  # self.errors in {"NaN", "ignore"}
            fps = []
            for mol in X:
                try:
                    fp = self._get_fp(mol)
                except ValueError:
                    fp = np.full(self.n_features_out, np.NaN)
                fps.append(fp)

        return np.array(fps)

    def _get_fp(self, mol: Mol) -> np.ndarray:
        conf_id = mol.GetIntProp("conf_id")
        coords = mol.GetConformer(conf_id).GetPositions()
        charges = self._get_atomic_charges(mol) * self.charge_scaling_factor
        descriptors = np.column_stack((coords, charges))
        centroid_dists = self._get_centroid_distances(descriptors, charges)

        fp = []
        for d in centroid_dists:
            # three moments: mean, stddev, cubic root of skewness
            fp.extend([np.mean(d), np.std(d), np.cbrt(moment(d, 3))])

        return np.array(fp)

    def _get_atomic_charges(self, mol: Mol) -> np.ndarray:
        from rdkit.Chem import MolToSmiles, rdPartialCharges
        from rdkit.Chem.rdForceFieldHelpers import MMFFGetMoleculeProperties

        atoms = mol.GetAtoms()

        if self.partial_charge_model == "Gasteiger":
            rdPartialCharges.ComputeGasteigerCharges(mol)
            charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in atoms]
        elif self.partial_charge_model == "MMFF94":
            values = MMFFGetMoleculeProperties(mol)
            charges = [
                values.GetMMFFPartialCharge(i) if values else None
                for i in range(len(atoms))
            ]
        elif self.partial_charge_model == "formal":
            charges = [atom.GetFormalCharge() for atom in atoms]
        else:  # precomputed
            charges = [atom.GetDoubleProp("charge") for atom in atoms]

        if self.charge_errors == "raise" and any(charge is None for charge in charges):
            smiles = MolToSmiles(mol)
            raise ValueError(
                f"Failed to compute at least one atom partial charge for {smiles}"
            )
        elif self.charge_errors == "zero":
            charges = np.nan_to_num(np.array(charges, dtype=float), nan=0)
        else:  # "ignore"
            charges = [c for c in charges if c is not None]

        return np.asarray(charges, dtype=float)

    def _get_centroid_distances(
        self, descriptors: np.ndarray, charges: np.ndarray
    ) -> list[np.ndarray]:
        # geometric center
        c1 = descriptors.mean(axis=0)
        dists_c1 = norm(descriptors - c1, axis=1)

        # furthest atom from c1
        c2 = descriptors[dists_c1.argmax()]
        dists_c2 = norm(descriptors - c2, axis=1)

        # furthest atom from c2
        c3 = descriptors[dists_c2.argmax()]
        dists_c3 = norm(descriptors - c3, axis=1)

        # vectors between centroids
        vec_a = c2 - c1
        vec_b = c3 - c1

        # scaled vector product of spatial coordinates (a_S and b_S)
        # it distinguishes between a chiral molecule and its enantiomer
        cross_ab = np.cross(vec_a[:3], vec_b[:3])
        cross_ab_norm = norm(cross_ab)
        if np.isclose(cross_ab_norm, 0):
            vec_c = 0
        else:
            vec_c = (norm(vec_a) / (2 * cross_ab_norm)) * cross_ab

        # geometric mean centroid moved in the direction of smallest and largest charge
        # note that charges were already scaled before
        c4 = np.append(c1[:3] + vec_c, np.max(charges))
        c5 = np.append(c1[:3] + vec_c, np.min(charges))

        dists_c4 = norm(descriptors - c4, axis=1)
        dists_c5 = norm(descriptors - c5, axis=1)

        return [dists_c1, dists_c2, dists_c3, dists_c4, dists_c5]
