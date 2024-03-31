from typing import Optional, Sequence, Union

import numpy as np
import scipy.sparse
from e3fp.conformer.generate import (
    FORCEFIELD_DEF,
    MAX_ENERGY_DIFF_DEF,
    POOL_MULTIPLIER_DEF,
    RMSD_CUTOFF_DEF,
)
from e3fp.conformer.generator import ConformerGenerator
from e3fp.pipeline import fprints_from_mol
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer
from skfp.validators import ensure_smiles

"""
Note: this file cannot have the "e3fp.py" name due to conflict with E3FP library.
"""


class E3FPFingerprint(FingerprintTransformer):
    """E3FP fingerprint."""

    def __init__(
        self,
        fp_size: int = 1024,
        n_bits_before_hash: int = 4096,
        radius_multiplier: float = 1.5,
        rdkit_invariants: bool = True,
        num_conf_generated: int = 3,
        num_conf_used: int = 1,
        pool_multiplier: float = POOL_MULTIPLIER_DEF,
        rmsd_cutoff: float = RMSD_CUTOFF_DEF,
        max_energy_diff: float = MAX_ENERGY_DIFF_DEF,
        force_field: float = FORCEFIELD_DEF,
        get_values: bool = True,
        aggregation_type: str = "min_energy",
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: int = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            n_jobs=n_jobs,
            verbose=verbose,
            sparse=sparse,
            random_state=random_state,
        )
        self.fp_size = fp_size
        self.n_bits_before_hash = n_bits_before_hash
        self.radius_multiplier = radius_multiplier
        self.rdkit_invariants = rdkit_invariants
        self.num_conf_generated = num_conf_generated
        self.num_conf_used = num_conf_used
        self.pool_multiplier = pool_multiplier
        self.rmsd_cutoff = rmsd_cutoff
        self.max_energy_diff = max_energy_diff
        self.force_field = force_field
        self.get_values = get_values
        self.aggregation_type = aggregation_type

    def _calculate_fingerprint(self, X: Sequence[str]) -> Union[np.ndarray, csr_array]:
        X = ensure_smiles(X)
        X = [self._calculate_single_mol_fingerprint(smi) for smi in X]
        return scipy.sparse.vstack(X) if self.sparse else np.array(X)

    def _calculate_single_mol_fingerprint(
        self, smiles: str
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem import MolFromSmiles
        from rdkit.Chem.PropertyMol import PropertyMol

        conf_gen = ConformerGenerator(
            first=self.num_conf_used,
            num_conf=self.num_conf_generated,
            pool_multiplier=self.pool_multiplier,
            rmsd_cutoff=self.rmsd_cutoff,
            max_energy_diff=self.max_energy_diff,
            forcefield=self.force_field,
            get_values=self.get_values,
            seed=self.random_state,
        )

        mol = MolFromSmiles(smiles)
        mol.SetProp("_Name", smiles)
        mol = PropertyMol(mol)
        mol.SetProp("_SMILES", smiles)

        # Generating conformers
        # TODO: for some molecules conformers are not properly generated - returns an empty list and throws RuntimeError
        try:
            mol, values = conf_gen.generate_conformers(mol)
            fps = fprints_from_mol(
                mol,
                fprint_params={
                    "bits": self.n_bits_before_hash,
                    "radius_multiplier": self.radius_multiplier,
                    "rdkit_invariants": self.rdkit_invariants,
                },
            )

            # TODO: add other aggregation types
            if self.aggregation_type == "min_energy":
                energies = values[2]
                fp = fps[np.argmin(energies)]
            else:
                fp = fps[0]

            fp = fp.fold(self.fp_size)
            return fp.to_vector(sparse=self.sparse)
        except RuntimeError:
            fp = np.full(shape=self.fp_size, fill_value=-1)
            return csr_array(fp) if self.sparse else fp
