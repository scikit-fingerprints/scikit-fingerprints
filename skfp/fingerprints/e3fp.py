from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse as spsparse
from e3fp.conformer.generate import (
    FORCEFIELD_DEF,
    MAX_ENERGY_DIFF_DEF,
    NUM_CONF_DEF,
    POOL_MULTIPLIER_DEF,
    RMSD_CUTOFF_DEF,
)
from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles
from rdkit.Chem.PropertyMol import PropertyMol

from skfp.fingerprints.base import FingerprintTransformer


class E3FP(FingerprintTransformer):
    def __init__(
        self,
        bits: int = 4096,
        radius_multiplier: float = 1.5,
        rdkit_invariants: bool = True,
        first: int = 1,
        num_conf: int = NUM_CONF_DEF,
        pool_multiplier: float = POOL_MULTIPLIER_DEF,
        rmsd_cutoff: float = RMSD_CUTOFF_DEF,
        max_energy_diff: float = MAX_ENERGY_DIFF_DEF,
        force_field: float = FORCEFIELD_DEF,
        get_values: bool = True,
        is_folded: bool = False,
        fold_bits: int = 1024,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
        random_state: int = 0,
        aggregation_type: str = "min_energy",
        count: bool = False,  # unused
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
            sparse=sparse,
            random_state=random_state,
            count=count,
        )
        self.bits = bits
        self.radius_multiplier = radius_multiplier
        self.rdkit_invariants = rdkit_invariants
        self.first = first
        self.num_conf = num_conf
        self.pool_multiplier = pool_multiplier
        self.rmsd_cutoff = rmsd_cutoff
        self.max_energy_diff = max_energy_diff
        self.force_field = force_field
        self.get_values = get_values
        self.is_folded = is_folded
        self.fold_bits = fold_bits
        self.aggregation_type = aggregation_type

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        from e3fp.conformer.generator import ConformerGenerator
        from e3fp.pipeline import fprints_from_mol

        conf_gen = ConformerGenerator(
            first=self.first,
            num_conf=self.num_conf,
            pool_multiplier=self.pool_multiplier,
            rmsd_cutoff=self.rmsd_cutoff,
            max_energy_diff=self.max_energy_diff,
            forcefield=self.force_field,
            get_values=self.get_values,
            seed=self.random_state,
        )

        def e3fp_function(x):
            if isinstance(x, Mol):
                smiles = MolToSmiles(x)
                mol = x
            else:
                smiles = x
                mol = MolFromSmiles(x)

            mol.SetProp("_Name", smiles)
            mol = PropertyMol(mol)
            mol.SetProp("_SMILES", smiles)
            # Generating conformers. Only few first conformers with lowest energy are used - specified by self.first
            # TODO: it appears, that for some molecules conformers are not properly generated - returns an empty list
            #  and throws RuntimeError
            try:
                mol, values = conf_gen.generate_conformers(mol)
                fps = fprints_from_mol(
                    mol,
                    fprint_params={
                        "bits": self.bits,
                        "radius_multiplier": self.radius_multiplier,
                        "rdkit_invariants": self.rdkit_invariants,
                    },
                )

                # TODO: in future - add other aggregation types
                if self.aggregation_type == "min_energy":
                    energies = values[2]
                    fp = fps[np.argmin(energies)]
                else:
                    fp = fps[0]

                if self.is_folded:
                    fp = fp.fold(self.fold_bits)

                return fp.to_vector()
            except RuntimeError:
                return spsparse.csr_array(
                    np.full(shape=self.bits, fill_value=-1)
                )

        result = [e3fp_function(x) for x in X]
        if self.sparse:
            return spsparse.vstack(result)
        else:
            return np.array([fp.toarray().squeeze() for fp in result])
