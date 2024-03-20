from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse
from rdkit.Chem import AddHs, Mol
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer
from skfp.validators import ensure_mols


class PubChemFingerprint(FingerprintTransformer):
    def __init__(
        self,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        X = ensure_mols(X)

        X = [self._get_pubchem_fingerprint(x) for x in X]
        return scipy.sparse.vstack(X) if self.sparse else np.vstack(X)

    def _get_pubchem_fingerprint(self, mol: Mol) -> Union[np.ndarray, csr_array]:
        # PubChem definition requires hydrogens to be present
        mol = AddHs(mol)

        # atom_counts = self._get_atom_counts(mol)
        # ring_counts = self._get_ESSSR_ring_counts(mol)
        # atom_pair_counts = self._get_atom_pair_counts(mol)
        # simple_neigh_counts = self._get_simple_neighborhoods_counts(mol)
        # detailed_neigh_counts = self._get_detailed_neighborhoods_counts(mol)
        # simple_smarts_counts = self._get_simple_smarts_patterns_counts(mol)
        # complex_smarts_counts = self._get_complex_smarts_patterns_counts(mol)

        return np.array([])

    def _get_atom_counts(self, mol: Mol) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for atom in mol.GetAtoms():
            counts[atom.GetSymbol()] += 1
        return counts

    def _get_ESSSR_ring_counts(self, mol: Mol) -> Dict[str, int]:
        # map
        counts: Dict[str, int] = defaultdict(int)

        ring_info = mol.GetRingInfo()
        for ring in ring_info.BondRings():
            ring = self._get_ring_stats(mol, ring)
            size = ring["size"]

            counts[f"ring_size_{size}"] += 1
            counts[f"ring_saturated_or_aromatic_carbon_only_{size}"] += (
                ring["saturated_or_aromatic"] & ring["carbon_only"]
            )
            counts[f"ring_saturated_or_aromatic_contains_nitrogen_{size}"] += (
                ring["saturated_or_aromatic"] & ring["contains_nitrogen"]
            )
            counts[f"ring_saturated_or_aromatic_contains_heteroatom_{size}"] += (
                ring["saturated_or_aromatic"] & ring["contains_heteroatom"]
            )
            counts[f"ring_unsaturated_and_non_aromatic_carbon_only_{size}"] += (
                ring["unsaturated_and_non_aromatic"] & ring["carbon_only"]
            )
            counts[f"ring_unsaturated_and_non_aromatic_contains_nitrogen_{size}"] += (
                ring["unsaturated_and_non_aromatic"] & ring["contains_nitrogen"]
            )
            counts[f"ring_unsaturated_and_non_aromatic_contains_heteroatom_{size}"] += (
                ring["unsaturated_and_non_aromatic"] & ring["contains_heteroatom"]
            )
            counts[f"ring_aromatic_{size}"] += ring["aromatic"]
            counts[f"ring_hetero_aromatic_{size}"] += ring["hetero_aromatic"]

        return counts

    def _get_ring_stats(
        self, mol: Mol, ring: Tuple[int]
    ) -> Dict[str, Union[int, bool]]:
        from rdkit.Chem import BondType

        stats = {
            "size": 0,
            "aromatic": True,
            "hetero_aromatic": True,
            "saturated_or_aromatic": True,
            "unsaturated_and_non_aromatic": True,
            "carbon_only": True,
            "contains_nitrogen": False,
            "contains_heteroatom": False,
        }

        for bond_idx in ring:
            bond = mol.GetBondWithIdx(bond_idx)
            atom_1_type = bond.GetBeginAtom().GetSymbol()
            atom_2_type = bond.GetEndAtom().GetSymbol()

            single = bond.GetBondType() == BondType.SINGLE
            aromatic = bond.GetIsAromatic()

            stats["size"] += 1
            stats["aromatic"] &= aromatic
            stats["saturated_or_aromatic"] &= single | aromatic
            stats["unsaturated_and_non_aromatic"] &= ~single & ~aromatic
            stats["carbon_only"] &= (atom_1_type == "C") & (atom_2_type == "C")
            stats["contains_nitrogen"] |= (atom_1_type == "N") | (atom_2_type == "N")
            stats["contains_heteroatom"] |= (atom_1_type == "C") | (atom_2_type == "H")

        stats["hetero_aromatic"] = ~stats["carbon_only"] & stats["aromatic"]

        return stats

    def _get_atom_pair_counts(self, mol: Mol) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for bond in mol.GetBonds():
            atom_1_type = bond.GetBeginAtom().GetSymbol()
            atom_2_type = bond.GetEndAtom().GetSymbol()
            counts[f"{atom_1_type}-{atom_2_type}"] += 1
        return counts
