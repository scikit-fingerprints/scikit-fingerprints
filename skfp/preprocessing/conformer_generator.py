from collections.abc import Sequence
from copy import deepcopy
from numbers import Integral
from typing import Optional

import numpy as np
from joblib import effective_n_jobs
from rdkit import Chem
from rdkit.Chem import AddHs, Mol, MolToSmiles, RemoveHs
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
from sklearn.utils import Interval
from sklearn.utils._param_validation import StrOptions

from skfp.parallel import run_in_parallel
from skfp.preprocessing.base import BasePreprocessor


class ConformerGenerator(BasePreprocessor):
    _parameter_constraints: dict = {
        "num_conformers": [Interval(Integral, 1, None, closed="left")],
        "max_gen_attempts": [Interval(Integral, 1, None, closed="left")],
        "error_on_conf_gen_fail": ["boolean"],
        "optimize_force_field": [StrOptions({"UFF", "MMFF94", "MMFF94s"}), None],
        "multiple_confs_select": [StrOptions({"min_energy", "first"})],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        num_conformers: int = 1,
        max_gen_attempts: int = 10000,
        error_on_gen_fail: bool = True,
        optimize_force_field: Optional[str] = None,
        multiple_confs_select: Optional[str] = "min_energy",
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[int] = 0,
    ):
        self.num_conformers = num_conformers
        self.max_gen_attempts = max_gen_attempts
        self.error_on_gen_fail = error_on_gen_fail
        self.optimize_force_field = optimize_force_field
        self.multiple_confs_select = multiple_confs_select
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state

        # make sure that conf_id property gets saved when pickle is used, e.g. for
        # parallelism with Joblib; this should be set every time this class is used
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.MolProps)

    def transform_x_y(
        self, X: Sequence[Mol], y: np.ndarray, copy: bool = False
    ) -> tuple[Sequence[Mol], np.ndarray]:
        return self._transform(X, y, copy)

    def transform(self, X: Sequence[Mol], copy: bool = False) -> Sequence[Mol]:
        y = np.zeros(len(X))
        X, y = self._transform(X, y, copy)
        return X

    def _transform(
        self, X: Sequence[Mol], y: np.ndarray, copy: bool = False
    ) -> tuple[Sequence[Mol], np.ndarray]:
        self._validate_params()

        if copy:
            X = deepcopy(X)
            y = deepcopy(y)

        n_jobs = effective_n_jobs(self.n_jobs)
        if n_jobs == 1:
            conformer_ids = self._embed_molecules(X)
        else:
            conformer_ids = run_in_parallel(
                self._embed_molecules,
                data=X,
                n_jobs=n_jobs,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )

        conf_generated_idxs = []
        for idx, (mol, conf_id) in enumerate(zip(X, conformer_ids)):
            mol.SetIntProp("conf_id", conf_id)
            if conf_id == -1:
                conf_generated_idxs.append(idx)

        # keep labels only for molecules for which we generated conformers
        y = y[conf_generated_idxs]

        return X, y

    def _embed_molecules(self, mols: Sequence[Mol]) -> list[int]:
        conf_ids = [self._embed_molecule(mol) for mol in mols]

        if self.optimize_force_field is not None:
            for mol in mols:
                self._optimize_conformers(mol)

        if self.num_conformers > 1:
            conf_ids = [self._select_conformer(mol) for mol in mols]

        return conf_ids

    def _embed_molecule(self, mol: Mol) -> int:
        # adding hydrogens is recommended for conformer generation
        mol = AddHs(mol)
        Chem.SanitizeMol(mol)

        # we create a new embedding params for each molecule, since it can
        # get modified if default settings fail to generate conformers
        embed_params = ETKDGv3()
        embed_params.useSmallRingTorsions = True
        embed_params.trackFailures = True
        embed_params.randomSeed = self.random_state

        if self.num_conformers == 1:
            embedder = EmbedMolecule
        else:
            embedder = EmbedMultipleConfs

        # basic attempt
        conf_id = embedder(mol, embed_params)

        if conf_id == -1:
            # more tries
            embed_params.maxIterations = self.max_gen_attempts
            embed_params.useRandomCoords = True
            conf_id = embedder(mol, embed_params)

        if conf_id == -1:
            # turn off conditions
            embed_params.enforceChirality = False
            embed_params.ignoreSmoothingFailures = True
            conf_id = embedder(mol, embed_params)

        # we should not fail at this point
        if conf_id == -1:
            smiles = MolToSmiles(RemoveHs(mol))
            fail_reason = self._print_conf_gen_failure_reason(embed_params)
            if self.error_on_gen_fail:
                raise ValueError(
                    f"Could not generate conformer for {smiles}:\n{fail_reason}"
                )
            elif self.verbose:
                print(f"Could not generate conformer for {smiles}:\n{fail_reason}")

        return conf_id

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
