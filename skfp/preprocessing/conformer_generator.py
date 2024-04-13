from collections.abc import Sequence
from numbers import Integral
from typing import Optional

from joblib import effective_n_jobs
from rdkit import Chem
from rdkit.Chem import AddHs, Mol, MolToSmiles, RemoveHs
from rdkit.Chem.rdDistGeom import EmbedFailureCauses, EmbedMolecule, ETKDGv3
from sklearn.utils import Interval

from skfp.parallel import run_in_parallel
from skfp.preprocessing.base import BasePreprocessor


class ConformerGenerator(BasePreprocessor):
    _parameter_constraints: dict = {
        "max_conf_gen_attempts": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        max_conf_gen_attempts: int = 10000,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[int] = 0,
    ):
        self.max_conf_gen_attempts = max_conf_gen_attempts
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def transform(self, X: Sequence[Mol], copy: bool = False) -> list[Mol]:
        self._validate_params()

        # adding hydrogens is recommended for conformer generation
        X = [AddHs(mol) for mol in X]

        n_jobs = effective_n_jobs(self.n_jobs)
        if n_jobs == 1:
            conformer_ids = self._embed_molecules(X)
        else:
            conformer_ids = run_in_parallel(
                self._embed_molecules, data=X, n_jobs=n_jobs, verbose=self.verbose
            )

        X = [RemoveHs(mol) for mol in X]

        # make sure that conf_id property gets saved when pickle is used,
        # e.g. for parallelism with Joblib
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.MolProps)

        for mol, conf_id in zip(X, conformer_ids):
            mol.SetIntProp("conf_id", conf_id)

        return X

    def _embed_molecules(self, mols: list[Mol]) -> list[int]:
        return [self._embed_molecule(mol) for mol in mols]

    def _embed_molecule(self, mol: Mol) -> int:
        # we create a new embedding params for each molecule, since it can
        # get modified if default settings fail to generate conformers
        embed_params = ETKDGv3()
        embed_params.trackFailures = True
        embed_params.randomSeed = self.random_state

        # basic attempt
        conf_id = EmbedMolecule(mol, embed_params)

        if conf_id == -1:
            # more tries
            embed_params.maxIterations = self.max_conf_gen_attempts
            embed_params.useRandomCoords = True
            conf_id = EmbedMolecule(mol, embed_params)

        if conf_id == -1:
            # turn off conditions
            embed_params.enforceChirality = False
            embed_params.ignoreSmoothingFailures = True
            conf_id = EmbedMolecule(mol, embed_params)

        # we should not fail at this point
        if conf_id == -1:
            smiles = MolToSmiles(RemoveHs(mol))
            fail_reason = self._print_conf_gen_failure_reason(embed_params)
            raise ValueError(
                f"Could not generate conformer for {smiles}:\n{fail_reason}"
            )

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
