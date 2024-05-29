from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from numbers import Integral
from typing import Optional

import numpy as np
from joblib import effective_n_jobs
from rdkit.Chem import AddHs, Mol, MolToSmiles, RemoveHs, SanitizeMol
from rdkit.Chem.PropertyMol import PropertyMol
from rdkit.Chem.rdDistGeom import (
    EmbedFailureCauses,
    EmbedMolecule,
    EmbedMultipleConfs,
    ETKDGv3,
)
from rdkit.Chem.rdForceFieldHelpers import (
    MMFFGetMoleculeForceField,
    MMFFGetMoleculeProperties,
    MMFFSanitizeMolecule,
    UFFGetMoleculeForceField,
)
from rdkit.ForceField import ForceField
from sklearn.utils._param_validation import Interval, InvalidParameterError, StrOptions

from skfp.bases import BasePreprocessor
from skfp.parallel import run_in_parallel
from skfp.validators import ensure_mols


class ConformerGenerator(BasePreprocessor):
    """
    Generate molecule conformer.

    The implementation uses RDKit and distance geometry (DG) approach [1], with
    optimized ETKDGv3 improvements for small rings, macrocycles and experimental
    torsional angle preferences [2]. Generated conformations are optionally optimized
    with a force field approach.

    Resulting conformation is saved in ``conf_id`` integer property of a molecule, and
    can be retrieved with ``GetIntProp("conf_id")`` method.

    If multiple conformations are generated, one per molecule is returned. By default,
    the most stable conformer (with the lowest energy) is selected.

    Note that conformer generation can fail, either due to not enough iterations, or
    it can be just impossible for a given molecule [3]. This by default results in an
    error, but can be controlled with ``error_on_gen_fail`` parameter. For multiple
    conformers, error is thrown only when no conformations can be generated, not if
    any one fails.

    If ``error_on_gen_fail`` is False and no conformers could be generated, the number
    of returned molecules will be smaller than input length. For supervised learning,
    use ``transform_x_y()`` instead of ``transform`` method to properly return labels
    for those molecules.

    Parameters
    ----------
    num_conformers : int, default=1
        Number of conformers to initially generate for each molecule. Must be positive.
        If larger than 1, one conformation will be selected, as specified with
        ``multiple_confs_select``.

    max_gen_attempts : int, default=10000
        Number of attempts to generate a conformer. Must be positive, and should be
        sufficiently large, typically at least a few thousands.

    optimize_force_field : {"UFF", "MMFF94", "MMFF94s", None}, default=None
        Force field optimization algorithm used on generated conformers. It is also
        used for calculation of conformer energy when selecting one of multiple
        conformations with ``multiple_confs_select="min_energy"``.

    multiple_confs_select : {"min_energy", "first"}, default="min_energy"
        How to select final conformer for each molecule when multiple conformers are
        generated. "first" selects first conformer generated by RDKit.

    errors : {"raise", "ignore", "filter"}, default="raise"
        How to handle errors during conformer generation, if after all attempts and
        approaches a conformer could not be generated. ``"raise"`` immediately raises
        any errors, with failure reason. ``"ignore"`` suppresses errors and returns
        molecules without conformers and ``conf_id`` property set to `-1`. ``"filter"``
        suppresses errors and removes such conformers, resulting in potentially less
        output molecules than inputs. The latter two options can potentially cause
        problems downstream, and should be used with caution.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform_x_y` and
        :meth:`transform` are parallelized over the input molecules. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See Scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int, default=0
        Controls the verbosity when generating conformers.

    random_state : int or None, default=0
        Controls the randomness of conformer generation. Note that in constrast to
        most classes, here it cannot be a ``RandomState`` instance, only an integer.

    References
    ----------
    .. [1] `Jean-Paul Ebejer
        "Conformer Generation using RDKit"
        1st RDKit User General Meeting, London, 2012
        <https://www.rdkit.org/UGM/2012/Ebejer_20110926_RDKit_1stUGM.pdf>`_

    .. [2] `Shuzhe Wang, Jagna Witek, Gregory A. Landrum, and Sereina Riniker
        "Improving Conformer Generation for Small Rings and Macrocycles Based on
        Distance Geometry and Experimental Torsional-Angle Preferences"
        J. Chem. Inf. Model. 2020, 60, 4, 2044–2058
        <https://pubs.acs.org/doi/10.1021/acs.jcim.0c00025>`_

    .. [3] `Gregory A. Landrum
        "Understanding conformer generation failures"
        RDKit blog, 2023
        <https://greglandrum.github.io/rdkit-blog/posts/2023-05-17-understanding-confgen-errors.html>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolFromSmilesTransformer, ConformerGenerator
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> conf_gen = ConformerGenerator()
    >>> conf_gen
    ConformerGenerator()

    >>> mol_from_smiles = MolFromSmilesTransformer()
    >>> mols = mol_from_smiles.transform(smiles)
    >>> conf_gen.transform(mols)
    [<rdkit.Chem.rdchem.Mol,
    <rdkit.Chem.rdchem.Mol,
    <rdkit.Chem.rdchem.Mol,
    <rdkit.Chem.rdchem.Mol]
    """

    _parameter_constraints: dict = {
        "num_conformers": [Interval(Integral, 1, None, closed="left")],
        "max_gen_attempts": [Interval(Integral, 1, None, closed="left")],
        "error_on_conf_gen_fail": ["boolean"],
        "optimize_force_field": [StrOptions({"UFF", "MMFF94", "MMFF94s"}), None],
        "multiple_confs_select": [StrOptions({"min_energy", "first"})],
        "errors": [StrOptions({"raise", "ignore", "filter"})],
        "n_jobs": [Integral, None],
        "batch_size": [Integral, None],
        "verbose": ["verbose"],
        "random_state": [Interval(Integral, left=-1, right=None, closed="left")],
    }

    def __init__(
        self,
        num_conformers: int = 1,
        max_gen_attempts: int = 1000,
        optimize_force_field: Optional[str] = None,
        multiple_confs_select: Optional[str] = "min_energy",
        errors: str = "raise",
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[int] = 0,
    ):
        self.num_conformers = num_conformers
        self.max_gen_attempts = max_gen_attempts
        self.optimize_force_field = optimize_force_field
        self.multiple_confs_select = multiple_confs_select
        self.errors = errors
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state

    def _validate_params(self) -> None:
        super()._validate_params()
        if (
            self.num_conformers > 1
            and self.multiple_confs_select == "min_energy"
            and self.optimize_force_field is None
        ):
            raise InvalidParameterError(
                "For selecting one of multiple conformers with lowest energy "
                '(num_conformers > 1 and multiple_confs_select == "min_energy"), '
                "force field optimization algorithm must be selected with "
                "optimize_force_field parameter, got None."
            )

    def transform_x_y(
        self, X: Sequence[Mol], y: np.ndarray, copy: bool = False
    ) -> tuple[list[PropertyMol], np.ndarray]:
        """
        Generate conformers for molecules.

        If ``errors`` is set to ``"filter"``, then in case of errors less than
        n_samples values may be returned. If ``errors`` is set to ``"ignore"``,
        some molecules may be returned without conformers generated and with
        ``conf_id`` property set to -1.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit Mol objects.

        y : np.ndarray of shape (n_samples,)
            Array with labels for molecules.

        copy : bool, default=True
            Copy the input X or not. In contrast to most classes, input molecules
            are copied by default, since RDKit modifies them with conformers in place.

        Returns
        -------
        X : list of shape (n_samples_conf_gen,)
            List with RDKit PropertyMol objects, each one with conformers computed and
            ``conf_id`` integer property set.

        y : np.ndarray of shape (n_samples_conf_gen,)
            Array with labels for molecules.
        """
        return self._transform(X, y, copy)

    def transform(self, X: Sequence[Mol], copy: bool = False) -> list[PropertyMol]:
        """
        Generate conformers for molecules.

        If ``errors`` is set to ``"filter"``, then in case of errors less than
        n_samples values may be returned. If ``errors`` is set to ``"ignore"``,
        some molecules may be returned without conformers generated and with
        ``conf_id`` property set to -1.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit Mol objects.

        copy : bool, default=True
            Copy the input X or not. In contrast to most classes, input molecules
            are copied by default, since RDKit modifies them with conformers in place.

        Returns
        -------
        X : list of shape (n_samples_conf_gen,)
            List with RDKit PropertyMol objects, each one with conformers computed and
            ``conf_id`` integer property set.
        """
        y = np.empty(len(X))
        X, y = self._transform(X, y, copy)
        return X

    def _transform(
        self, X: Sequence[Mol], y: np.ndarray, copy: bool = True
    ) -> tuple[list[PropertyMol], np.ndarray]:
        self._validate_params()

        if copy:
            X = deepcopy(X)
            y = deepcopy(y)

        X = ensure_mols(X)

        n_jobs = effective_n_jobs(self.n_jobs)
        if n_jobs == 1:
            mols = self._embed_molecules(X)
        else:
            mols = run_in_parallel(
                self._embed_molecules,
                data=X,
                n_jobs=n_jobs,
                batch_size=self.batch_size,
                flatten_results=True,
                verbose=self.verbose,
            )

        # keep only molecules and labels for which we generated conformers
        mols_with_conformers = []
        idxs_to_keep = []
        for idx, mol in enumerate(mols):
            if mol.GetIntProp("conf_id") != -1 or self.errors == "ignore":
                mols_with_conformers.append(mol)
                idxs_to_keep.append(idx)

        y = y[idxs_to_keep]

        return mols_with_conformers, y

    def _embed_molecules(self, mols: Sequence[Mol]) -> list[Mol]:
        # adding hydrogens and sanitizing is recommended for conformer generation
        mols = [AddHs(mol) for mol in mols]
        for mol in mols:
            SanitizeMol(mol)

        # make sure properties are pickled properly
        mols = [PropertyMol(mol) for mol in mols]

        mols_and_conf_ids = [self._embed_molecule(mol) for mol in mols]

        if self.num_conformers > 1:
            mols_and_conf_ids = [
                (mol, self._select_conformer(mol)) for mol, conf_id in mols_and_conf_ids
            ]

        mols = []
        for mol, conf_id in mols_and_conf_ids:
            mol.SetIntProp("conf_id", conf_id)
            mols.append(mol)

        return mols

    def _embed_molecule(self, mol: Mol) -> tuple[Mol, int]:
        # we create a new embedding params for each molecule, since it can
        # get modified if default settings fail to generate conformers
        embed_params = ETKDGv3()
        embed_params.useSmallRingTorsions = True
        embed_params.trackFailures = True
        embed_params.randomSeed = self.random_state

        # basic attempt
        if self.num_conformers == 1:
            embedder = EmbedMolecule
        else:
            embedder = partial(EmbedMultipleConfs, numConfs=self.num_conformers)

        conf_id = embedder(mol, params=embed_params)

        if conf_id == -1:
            # more tries
            embed_params.maxIterations = self.max_gen_attempts
            embed_params.useRandomCoords = True
            conf_id = embedder(mol, params=embed_params)

        if conf_id == -1:
            # even more tries, turn off conditions
            embed_params.maxIterations = 10 * self.max_gen_attempts
            embed_params.enforceChirality = False
            embed_params.ignoreSmoothingFailures = True
            conf_id = embedder(mol, params=embed_params)

        # we should not fail at this point
        if conf_id == -1:
            smiles = MolToSmiles(RemoveHs(mol))
            fail_reason = self._print_conf_gen_failure_reason(embed_params)
            if self.errors == "raise":
                raise ValueError(
                    f"Could not generate conformer for {smiles}:\n{fail_reason}"
                )
            elif self.verbose:
                print(f"Could not generate conformer for {smiles}:\n{fail_reason}")
                return mol, -1

        if self.optimize_force_field:
            self._optimize_conformers(mol)

        return mol, conf_id

    def _print_conf_gen_failure_reason(self, embed_params: ETKDGv3) -> str:
        fail_idx_to_name = {idx: name for name, idx in EmbedFailureCauses.names.items()}
        fail_counts = embed_params.GetFailureCounts()
        fail_names_with_counts = [
            f"{fail_idx_to_name[idx]}: {fail_counts[idx]}"
            for idx in range(len(fail_counts))
        ]
        fail_reason = "\n".join(fail_names_with_counts)
        return fail_reason

    def _optimize_conformers(self, mol: Mol) -> None:
        for conf in mol.GetConformers():
            ff = self._get_force_field(mol, conf_id=conf.GetId())
            ff.Minimize()

    def _select_conformer(self, mol: Mol) -> int:
        if self.multiple_confs_select == "first":
            return next(mol.GetConformers())
        else:  # min_energy
            energies = np.empty((mol.GetNumConformers(),))
            for i, conf in enumerate(mol.GetConformers()):
                ff = self._get_force_field(mol, conf_id=conf.GetId())
                energies[i] = ff.CalcEnergy()
            return int(np.argmin(energies))

    def _get_force_field(self, mol: Mol, conf_id: int) -> ForceField:
        if self.optimize_force_field == "UFF":
            return UFFGetMoleculeForceField(mol, confId=conf_id)
        else:
            # MMFF94 or MMFF94s
            MMFFSanitizeMolecule(mol)
            mmff_props = MMFFGetMoleculeProperties(
                mol, mmffVariant=self.optimize_force_field
            )
            return MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
