from collections import defaultdict
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from rdkit.Chem import GetSymmSSSR, Mol
from scipy.sparse import csr_array

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class PubChemFingerprint(BaseFingerprintTransformer):
    """
    PubChem fingerprint, also known as CACTVS fingerprint.

    This is a custom implementation of descriptor-based PubChem substructure
    fingerprint, based on official PubChem definitions [1]_ and Chemistry Development
    Kit (CDK) implementation [2]_, including fixes proposed by Andrew Dalke [3]_.
    In particular, it works correctly with implicit hydrogens.

    Results can be slightly different from PubChem API due to usage of RDKit aromaticity
    model and symmetrized SSSR.

    Count version of this fingerprint uses counts instead of inequalities (e.g. count
    of carbons instead of C>=2, C>=4, ..., C>=32), and counts occurrences of SMARTS
    patterns (substructures) instead of only checking for their existence.

    Parameters
    ----------
    count : bool, default=False
        Whether to return binary (bit) features, or the count-based variant.

    sparse : bool, default=False
        Whether to return dense NumPy array, or sparse SciPy CSR array.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when computing fingerprints.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    Attributes
    ----------
    n_features_out : int = 881 or 757.
        Number of output features, size of fingerprints. Equal to 881 for the bit
        variant, and 757 for count.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    References
    ----------
    .. [1] `PubChem Substructure Fingerprint
        <https://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.txt>`_

    .. [2] `Chemistry Development Kit (CDK) PubchemFingerprinter
        <https://cdk.github.io/cdk/latest/docs/api/org/openscience/cdk/fingerprint/PubchemFingerprinter.html>`_

    .. [3] `Andrew Dalke
        "Implementing the CACTVS/PubChem substructure keys"
        <http://www.dalkescientific.com/writings/diary/archive/2011/01/20/implementing_cactvs_keys.html>`_

    Examples
    --------
    >>> from skfp.fingerprints import PubChemFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = PubChemFingerprint()
    >>> fp
    PubChemFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 0],
           [1, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [1, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    """

    def __init__(
        self,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        n_features_out = 757 if count else 881
        super().__init__(
            n_features_out=n_features_out,
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
        Compute PubChem fingerprints. Output shape depends on ``count``
        parameter.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.n_features_out)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        X = ensure_mols(X)

        X = [self._get_pubchem_fingerprint(mol) for mol in X]
        return csr_array(X) if self.sparse else np.vstack(X)

    def _get_pubchem_fingerprint(self, mol: Mol) -> Union[np.ndarray, csr_array]:
        atom_counts = self._get_atom_counts(mol)
        ring_counts = self._get_ESSSR_ring_counts(mol)
        atom_pair_counts = self._get_atom_pair_counts(mol)
        simple_neigh_counts = self._get_simple_neighborhoods_counts(mol)
        detailed_neigh_counts = self._get_detailed_neighborhoods_counts(mol)
        simple_smarts_counts = self._get_simple_smarts_patterns_counts(mol)
        complex_smarts_counts = self._get_complex_smarts_patterns_counts(mol)

        if not self.count:
            atom_features = [
                atom_counts["H"] >= 4,
                atom_counts["H"] >= 8,
                atom_counts["H"] >= 16,
                atom_counts["H"] >= 32,
                atom_counts["Li"] >= 1,
                atom_counts["Li"] >= 2,
                atom_counts["B"] >= 1,
                atom_counts["B"] >= 2,
                atom_counts["B"] >= 4,
                atom_counts["C"] >= 2,
                atom_counts["C"] >= 4,
                atom_counts["C"] >= 8,
                atom_counts["C"] >= 16,
                atom_counts["C"] >= 32,
                atom_counts["N"] >= 1,
                atom_counts["N"] >= 2,
                atom_counts["N"] >= 4,
                atom_counts["N"] >= 8,
                atom_counts["O"] >= 1,
                atom_counts["O"] >= 2,
                atom_counts["O"] >= 4,
                atom_counts["O"] >= 8,
                atom_counts["O"] >= 16,
                atom_counts["F"] >= 1,
                atom_counts["F"] >= 2,
                atom_counts["F"] >= 4,
                atom_counts["Na"] >= 1,
                atom_counts["Na"] >= 2,
                atom_counts["Si"] >= 1,
                atom_counts["Si"] >= 2,
                atom_counts["P"] >= 1,
                atom_counts["P"] >= 2,
                atom_counts["P"] >= 4,
                atom_counts["S"] >= 1,
                atom_counts["S"] >= 2,
                atom_counts["S"] >= 4,
                atom_counts["S"] >= 8,
                atom_counts["Cl"] >= 1,
                atom_counts["Cl"] >= 2,
                atom_counts["Cl"] >= 4,
                atom_counts["Cl"] >= 8,
                atom_counts["K"] >= 1,
                atom_counts["K"] >= 2,
                atom_counts["Br"] >= 1,
                atom_counts["Br"] >= 2,
                atom_counts["Br"] >= 4,
                atom_counts["I"] >= 1,
                atom_counts["I"] >= 2,
                atom_counts["I"] >= 4,
                atom_counts["Be"],
                atom_counts["Mg"] >= 1,
                atom_counts["Al"] >= 1,
                atom_counts["Ca"] >= 1,
                atom_counts["Sc"] >= 1,
                atom_counts["Ti"] >= 1,
                atom_counts["V"] >= 1,
                atom_counts["Cr"] >= 1,
                atom_counts["Mn"] >= 1,
                atom_counts["Fe"] >= 1,
                atom_counts["Co"] >= 1,
                atom_counts["Ni"] >= 1,
                atom_counts["Cu"] >= 1,
                atom_counts["Zn"] >= 1,
                atom_counts["Ga"] >= 1,
                atom_counts["Ge"] >= 1,
                atom_counts["As"] >= 1,
                atom_counts["Se"] >= 1,
                atom_counts["Kr"] >= 1,
                atom_counts["Rb"] >= 1,
                atom_counts["Sr"] >= 1,
                atom_counts["Y"] >= 1,
                atom_counts["Zr"] >= 1,
                atom_counts["Nb"] >= 1,
                atom_counts["Mo"] >= 1,
                atom_counts["Ru"] >= 1,
                atom_counts["Rh"] >= 1,
                atom_counts["Pd"] >= 1,
                atom_counts["Ag"] >= 1,
                atom_counts["Cd"] >= 1,
                atom_counts["In"] >= 1,
                atom_counts["Sn"] >= 1,
                atom_counts["Sb"] >= 1,
                atom_counts["Te"] >= 1,
                atom_counts["Xe"] >= 1,
                atom_counts["Cs"] >= 1,
                atom_counts["Ba"] >= 1,
                atom_counts["Lu"] >= 1,
                atom_counts["Hf"] >= 1,
                atom_counts["Ta"] >= 1,
                atom_counts["W"] >= 1,
                atom_counts["Re"] >= 1,
                atom_counts["Os"] >= 1,
                atom_counts["Ir"] >= 1,
                atom_counts["Pt"] >= 1,
                atom_counts["Au"] >= 1,
                atom_counts["Hg"] >= 1,
                atom_counts["Tl"] >= 1,
                atom_counts["Pb"] >= 1,
                atom_counts["Bi"] >= 1,
                atom_counts["La"] >= 1,
                atom_counts["Ce"] >= 1,
                atom_counts["Pr"] >= 1,
                atom_counts["Nd"] >= 1,
                atom_counts["Pm"] >= 1,
                atom_counts["Sm"] >= 1,
                atom_counts["Eu"] >= 1,
                atom_counts["Gd"] >= 1,
                atom_counts["Tb"] >= 1,
                atom_counts["Dy"] >= 1,
                atom_counts["Ho"] >= 1,
                atom_counts["Er"] >= 1,
                atom_counts["Tm"] >= 1,
                atom_counts["Yb"] >= 1,
                atom_counts["Tc"] >= 1,
                atom_counts["U"] >= 1,
            ]
            ring_features = self._get_ring_binary_features(ring_counts)
        else:
            atom_features = [
                atom_counts[atom_type]
                for atom_type in [
                    "H",
                    "Li",
                    "B",
                    "C",
                    "N",
                    "O",
                    "F",
                    "Na",
                    "Si",
                    "P",
                    "S",
                    "Cl",
                    "K",
                    "Br",
                    "I",
                    "Be",
                    "Mg",
                    "Al",
                    "Ca",
                    "Sc",
                    "Ti",
                    "V",
                    "Cr",
                    "Mn",
                    "Fe",
                    "Co",
                    "Ni",
                    "Cu",
                    "Zn",
                    "Ga",
                    "Ge",
                    "As",
                    "Se",
                    "Kr",
                    "Rb",
                    "Sr",
                    "Y",
                    "Zr",
                    "Nb",
                    "Mo",
                    "Ru",
                    "Rh",
                    "Pd",
                    "Ag",
                    "Cd",
                    "In",
                    "Sn",
                    "Sb",
                    "Te",
                    "Xe",
                    "Cs",
                    "Ba",
                    "Lu",
                    "Hf",
                    "Ta",
                    "W",
                    "Re",
                    "Os",
                    "Ir",
                    "Pt",
                    "Au",
                    "Hg",
                    "Tl",
                    "Pb",
                    "Bi",
                    "La",
                    "Ce",
                    "Pr",
                    "Nd",
                    "Pm",
                    "Sm",
                    "Eu",
                    "Gd",
                    "Tb",
                    "Dy",
                    "Ho",
                    "Er",
                    "Tm",
                    "Yb",
                    "Tc",
                    "U",
                ]
            ]
            ring_features = self._get_ring_count_features(ring_counts)

        X = np.array(
            atom_features
            + ring_features
            + atom_pair_counts
            + simple_neigh_counts
            + detailed_neigh_counts
            + simple_smarts_counts
            + complex_smarts_counts,
        )

        if self.count:
            return X.astype(np.uint32)
        else:
            return (X > 0).astype(np.uint8)

    def _get_atom_counts(self, mol: Mol) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for atom in mol.GetAtoms():
            counts[atom.GetSymbol()] += 1
            counts["H"] += atom.GetNumImplicitHs()  # implicit hydrogens

        return counts

    def _get_ESSSR_ring_counts(self, mol: Mol) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)

        for ring in GetSymmSSSR(mol):
            ring = self._get_ring_stats(mol, ring)
            size = ring["size"]

            counts[f"ring_size_{size}"] += 1

            # saturated or aromatic rings
            counts[f"ring_saturated_carbon_only_{size}"] += (
                ring["saturated_or_aromatic"] & ring["carbon_only"]
            )
            counts[f"ring_saturated_has_nitrogen_{size}"] += (
                ring["saturated_or_aromatic"] & ring["has_nitrogen"]
            )
            counts[f"ring_saturated_has_heteroatom_{size}"] += (
                ring["saturated_or_aromatic"] & ring["has_heteroatom"]
            )

            # unsaturated and non-aromatic rings
            counts[f"ring_unsaturated_carbon_only_{size}"] += (
                ring["unsaturated_and_non_aromatic"] & ring["carbon_only"]
            )
            counts[f"ring_unsaturated_has_nitrogen_{size}"] += (
                ring["unsaturated_and_non_aromatic"] & ring["has_nitrogen"]
            )
            counts[f"ring_unsaturated_has_heteroatom_{size}"] += (
                ring["unsaturated_and_non_aromatic"] & ring["has_heteroatom"]
            )

            counts[f"ring_aromatic_{size}"] += ring["aromatic"]
            counts[f"ring_hetero_aromatic_{size}"] += ring["hetero_aromatic"]

        return counts

    def _get_ring_stats(
        self, mol: Mol, ring: tuple[int]
    ) -> dict[str, Union[int, bool]]:
        from rdkit.Chem import BondType

        stats = {
            "size": 0,
            "aromatic": True,
            "hetero_aromatic": True,
            "saturated_or_aromatic": True,
            "unsaturated_and_non_aromatic": True,
            "carbon_only": True,
            "has_nitrogen": False,
            "has_heteroatom": False,
        }

        for i, atom_idx in enumerate(ring):
            atom_idx_2 = ring[i + 1] if i + 1 < len(ring) else ring[0]

            bond = mol.GetBondBetweenAtoms(atom_idx, atom_idx_2)
            atom_1_type = bond.GetBeginAtom().GetSymbol()
            atom_2_type = bond.GetEndAtom().GetSymbol()

            single = bond.GetBondType() == BondType.SINGLE
            aromatic = bond.GetIsAromatic()

            stats["size"] += 1
            stats["aromatic"] &= aromatic
            stats["saturated_or_aromatic"] &= single or aromatic
            stats["unsaturated_and_non_aromatic"] &= not single and not aromatic
            stats["carbon_only"] &= (atom_1_type == "C") and (atom_2_type == "C")
            stats["has_nitrogen"] |= (atom_1_type == "N") or (atom_2_type == "N")
            stats["has_heteroatom"] |= (atom_1_type == "C") or (atom_2_type == "H")

        stats["hetero_aromatic"] = not stats["carbon_only"] and stats["aromatic"]

        return stats

    def _get_ring_binary_features(self, ring_counts: dict[str, int]) -> list[int]:
        # for each ring size, we have different number of binary features, for multiple
        # count thresholds, e.g. for 7-atom rings we have 2 sets of features (>=1, >=2),
        # and for 5-atom rings we have 5 sets
        ring_size_count_thresholds = {
            3: [1, 2],
            4: [1, 2],
            5: [1, 2, 3, 4, 5],
            6: [1, 2, 3, 4, 5],
            7: [1, 2],
            8: [1, 2],
            9: [1],
            10: [1],
        }

        features = []

        num_aromatic_rings = 0
        num_hetero_aromatic_rings = 0

        for size in range(3, 11):
            for threshold in ring_size_count_thresholds[size]:
                ring_size_features = [
                    ring_counts[f"ring_size_{size}"] >= threshold,
                    ring_counts[f"ring_saturated_carbon_only_{size}"] >= threshold,
                    ring_counts[f"ring_saturated_has_nitrogen_{size}"] >= threshold,
                    ring_counts[f"ring_saturated_has_heteroatom_{size}"] >= threshold,
                    ring_counts[f"ring_unsaturated_carbon_only_{size}"] >= threshold,
                    ring_counts[f"ring_unsaturated_has_nitrogen_{size}"] >= threshold,
                    ring_counts[f"ring_unsaturated_has_heteroatom_{size}"] >= threshold,
                ]
                features.extend(ring_size_features)
                num_aromatic_rings += ring_counts[f"ring_aromatic_{size}"]
                num_hetero_aromatic_rings += ring_counts[f"ring_hetero_aromatic_{size}"]

        aromatic_ring_features = [
            num_aromatic_rings >= 1,
            num_hetero_aromatic_rings >= 1,
            num_aromatic_rings >= 2,
            num_hetero_aromatic_rings >= 2,
            num_aromatic_rings >= 3,
            num_hetero_aromatic_rings >= 3,
            num_aromatic_rings >= 4,
            num_hetero_aromatic_rings >= 4,
        ]
        features.extend(aromatic_ring_features)

        features = [int(feat) for feat in features]
        return features

    def _get_ring_count_features(self, ring_counts: dict[str, int]) -> list[int]:
        # for each ring size, we count all
        features = []

        num_aromatic_rings = 0
        num_hetero_aromatic_rings = 0

        for size in range(3, 11):
            ring_size_features = [
                ring_counts[f"ring_size_{size}"],
                ring_counts[f"ring_saturated_carbon_only_{size}"],
                ring_counts[f"ring_saturated_has_nitrogen_{size}"],
                ring_counts[f"ring_saturated_has_heteroatom_{size}"],
                ring_counts[f"ring_unsaturated_carbon_only_{size}"],
                ring_counts[f"ring_unsaturated_has_nitrogen_{size}"],
                ring_counts[f"ring_unsaturated_has_heteroatom_{size}"],
            ]
            features.extend(ring_size_features)
            num_aromatic_rings += ring_counts[f"ring_aromatic_{size}"]
            num_hetero_aromatic_rings += ring_counts[f"ring_hetero_aromatic_{size}"]

        features.append(num_aromatic_rings)
        features.append(num_hetero_aromatic_rings)

        return features

    def _get_atom_pair_counts(self, mol: Mol) -> list[int]:
        smarts_list = [
            "[Li&!H0]",
            "[Li]~[Li]",
            "[Li]~[#5]",
            "[Li]~[#6]",
            "[Li]~[#8]",
            "[Li]~[F]",
            "[Li]~[#15]",
            "[Li]~[#16]",
            "[Li]~[Cl]",
            "[#5&!H0]",
            "[#5]~[#5]",
            "[#5]~[#6]",
            "[#5]~[#7]",
            "[#5]~[#8]",
            "[#5]~[F]",
            "[#5]~[#14]",
            "[#5]~[#15]",
            "[#5]~[#16]",
            "[#5]~[Cl]",
            "[#5]~[Br]",
            "[#6&!H0]",
            "[#6]~[#6]",
            "[#6]~[#7]",
            "[#6]~[#8]",
            "[#6]~[F]",
            "[#6]~[Na]",
            "[#6]~[Mg]",
            "[#6]~[Al]",
            "[#6]~[#14]",
            "[#6]~[#15]",
            "[#6]~[#16]",
            "[#6]~[Cl]",
            "[#6]~[#33]",
            "[#6]~[#34]",
            "[#6]~[Br]",
            "[#6]~[I]",
            "[#7&!H0]",
            "[#7]~[#7]",
            "[#7]~[#8]",
            "[#7]~[F]",
            "[#7]~[#14]",
            "[#7]~[#15]",
            "[#7]~[#16]",
            "[#7]~[Cl]",
            "[#7]~[Br]",
            "[#8&!H0]",
            "[#8]~[#8]",
            "[#8]~[Mg]",
            "[#8]~[Na]",
            "[#8]~[Al]",
            "[#8]~[#14]",
            "[#8]~[#15]",
            "[#8]~[K]",
            "[F]~[#15]",
            "[F]~[#16]",
            "[Al&!H0]",
            "[Al]~[Cl]",
            "[#14&!H0]",
            "[#14]~[#14]",
            "[#14]~[Cl]",
            "[#15&!H0]",
            "[#15]~[#15]",
            "[#33&!H0]",
            "[#33]~[#33]",
        ]
        return self._get_smarts_match_counts(mol, smarts_list)

    def _get_simple_neighborhoods_counts(self, mol: Mol) -> list[int]:
        smarts_list = [
            "[#6](~Br)(~[#6])",
            "[#6](~Br)(~[#6])(~[#6])",
            "[#6&!H0]~[Br]",
            "[#6](~[Br])(:[c])",
            "[#6](~[Br])(:[n])",
            "[#6](~[#6])(~[#6])",
            "[#6](~[#6])(~[#6])(~[#6])",
            "[#6](~[#6])(~[#6])(~[#6])(~[#6])",
            "[#6H1](~[#6])(~[#6])(~[#6])",
            "[#6](~[#6])(~[#6])(~[#6])(~[#7])",
            "[#6](~[#6])(~[#6])(~[#6])(~[#8])",
            "[#6H1](~[#6])(~[#6])(~[#7])",
            "[#6H1](~[#6])(~[#6])(~[#8])",
            "[#6](~[#6])(~[#6])(~[#7])",
            "[#6](~[#6])(~[#6])(~[#8])",
            "[#6](~[#6])(~[Cl])",
            "[#6&!H0](~[#6])(~[Cl])",
            "[#6H,#6H2,#6H3,#6H4]~[#6]",
            "[#6&!H0](~[#6])(~[#7])",
            "[#6&!H0](~[#6])(~[#8])",
            "[#6H1](~[#6])(~[#8])(~[#8])",
            "[#6&!H0](~[#6])(~[#15])",
            "[#6&!H0](~[#6])(~[#16])",
            "[#6](~[#6])(~[I])",
            "[#6](~[#6])(~[#7])",
            "[#6](~[#6])(~[#8])",
            "[#6](~[#6])(~[#16])",
            "[#6](~[#6])(~[#14])",
            "[#6](~[#6])(:c)",
            "[#6](~[#6])(:c)(:c)",
            "[#6](~[#6])(:c)(:n)",
            "[#6](~[#6])(:n)",
            "[#6](~[#6])(:n)(:n)",
            "[#6](~[Cl])(~[Cl])",
            "[#6&!H0](~[Cl])",
            "[#6](~[Cl])(:c)",
            "[#6](~[F])(~[F])",
            "[#6](~[F])(:c)",
            "[#6&!H0](~[#7])",
            "[#6&!H0](~[#8])",
            "[#6&!H0](~[#8])(~[#8])",
            "[#6&!H0](~[#16])",
            "[#6&!H0](~[#14])",
            "[#6&!H0]:c",
            "[#6&!H0](:c)(:c)",
            "[#6&!H0](:c)(:n)",
            "[#6&!H0](:n)",
            "[#6H3]",
            "[#6](~[#7])(~[#7])",
            "[#6](~[#7])(:c)",
            "[#6](~[#7])(:c)(:c)",
            "[#6](~[#7])(:c)(:n)",
            "[#6](~[#7])(:n)",
            "[#6](~[#8])(~[#8])",
            "[#6](~[#8])(:c)",
            "[#6](~[#8])(:c)(:c)",
            "[#6](~[#16])(:c)",
            "[#6](:c)(:c)",
            "[#6](:c)(:c)(:c)",
            "[#6](:c)(:c)(:n)",
            "[#6](:c)(:n)",
            "[#6](:c)(:n)(:n)",
            "[#6](:n)(:n)",
            "[#7](~[#6])(~[#6])",
            "[#7](~[#6])(~[#6])(~[#6])",
            "[#7&!H0](~[#6])(~[#6])",
            "[#7&!H0](~[#6])",
            "[#7&!H0](~[#6])(~[#7])",
            "[#7](~[#6])(~[#8])",
            "[#7](~[#6])(:c)",
            "[#7](~[#6])(:c)(:c)",
            "[#7&!H0](~[#7])",
            "[#7&!H0](:c)",
            "[#7&!H0](:c)(:c)",
            "[#7](~[#8])(~[#8])",
            "[#7](~[#8])(:o)",
            "[#7](:c)(:c)",
            "[#7](:c)(:c)(:c)",
            "[#8](~[#6])(~[#6])",
            "[#8&!H0](~[#6])",
            "[#8](~[#6])(~[#15])",
            "[#8&!H0](~[#16])",
            "[#8](:c)(:c)",
            "[#15](~[#6])(~[#6])",
            "[#15](~[#8])(~[#8])",
            "[#16](~[#6])(~[#6])",
            "[#16&!H0](~[#6])",
            "[#16](~[#6])(~[#8])",
            "[#14](~[#6])(~[#6])",
        ]
        return self._get_smarts_match_counts(mol, smarts_list)

    def _get_detailed_neighborhoods_counts(self, mol: Mol) -> list[int]:
        smarts_list = [
            "[#6]=,:[#6]",
            "[#6]#[#6]",
            "[#6]=,:[#7]",
            "[#6]#[#7]",
            "[#6]=,:[#8]",
            "[#6]=,:[#16]",
            "[#7]=,:[#7]",
            "[#7]=,:[#8]",
            "[#7]=,:[#15]",
            "[#15]=,:[#8]",
            "[#15]=,:[#15]",
            "[#6](#[#6])(-,:[#6])",
            "[#6&!H0](#[#6])",
            "[#6](#[#7])(-,:[#6])",
            "[#6](-,:[#6])(-,:[#6])(=,:[#6])",
            "[#6](-,:[#6])(-,:[#6])(=,:[#7])",
            "[#6](-,:[#6])(-,:[#6])(=,:[#8])",
            "[#6](-,:[#6])([Cl])(=,:[#8])",
            "[#6&!H0](-,:[#6])(=,:[#6])",
            "[#6&!H0](-,:[#6])(=,:[#7])",
            "[#6&!H0](-,:[#6])(=,:[#8])",
            "[#6](-,:[#6])(-,:[#7])(=,:[#6])",
            "[#6](-,:[#6])(-,:[#7])(=,:[#7])",
            "[#6](-,:[#6])(-,:[#7])(=,:[#8])",
            "[#6](-,:[#6])(-,:[#8])(=,:[#8])",
            "[#6](-,:[#6])(=,:[#6])",
            "[#6](-,:[#6])(=,:[#7])",
            "[#6](-,:[#6])(=,:[#8])",
            "[#6]([Cl])(=,:[#8])",
            "[#6&!H0](-,:[#7])(=,:[#6])",
            "[#6&!H0](=,:[#6])",
            "[#6&!H0](=,:[#7])",
            "[#6&!H0](=,:[#8])",
            "[#6](-,:[#7])(=,:[#6])",
            "[#6](-,:[#7])(=,:[#7])",
            "[#6](-,:[#7])(=,:[#8])",
            "[#6](-,:[#8])(=,:[#8])",
            "[#7](-,:[#6])(=,:[#6])",
            "[#7](-,:[#6])(=,:[#8])",
            "[#7](-,:[#8])(=,:[#8])",
            "[#15](-,:[#8])(=,:[#8])",
            "[#16](-,:[#6])(=,:[#8])",
            "[#16](-,:[#8])(=,:[#8])",
            "[#16](=,:[#8])(=,:[#8])",
        ]
        return self._get_smarts_match_counts(mol, smarts_list)

    def _get_simple_smarts_patterns_counts(self, mol: Mol) -> list[int]:
        smarts_list = [
            "[#6]-,:[#6]-,:[#6]#[#6]",
            "[#8]-,:[#6]-,:[#6]=,:[#7]",
            "[#8]-,:[#6]-,:[#6]=,:[#8]",
            "[#7]:[#6]-,:[#16&!H0]",
            "[#7]-,:[#6]-,:[#6]=,:[#6]",
            "[#8]=,:[#16]-,:[#6]-,:[#6]",
            "[#7]#[#6]-,:[#6]=,:[#6]",
            "[#6]=,:[#7]-,:[#7]-,:[#6]",
            "[#8]=,:[#16]-,:[#6]-,:[#7]",
            "[#16]-,:[#16]-,:[#6]:[#6]",
            "[#6]:[#6]-,:[#6]=,:[#6]",
            "[#16]:[#6]:[#6]:[#6]",
            "[#6]:[#7]:[#6]-,:[#6]",
            "[#16]-,:[#6]:[#7]:[#6]",
            "[#16]:[#6]:[#6]:[#7]",
            "[#16]-,:[#6]=,:[#7]-,:[#6]",
            "[#6]-,:[#8]-,:[#6]=,:[#6]",
            "[#7]-,:[#7]-,:[#6]:[#6]",
            "[#16]-,:[#6]=,:[#7&!H0]",
            "[#16]-,:[#6]-,:[#16]-,:[#6]",
            "[#6]:[#16]:[#6]-,:[#6]",
            "[#8]-,:[#16]-,:[#6]:[#6]",
            "[#6]:[#7]-,:[#6]:[#6]",
            "[#7]-,:[#16]-,:[#6]:[#6]",
            "[#7]-,:[#6]:[#7]:[#6]",
            "[#7]:[#6]:[#6]:[#7]",
            "[#7]-,:[#6]:[#7]:[#7]",
            "[#7]-,:[#6]=,:[#7]-,:[#6]",
            "[#7]-,:[#6]=,:[#7&!H0]",
            "[#7]-,:[#6]-,:[#16]-,:[#6]",
            "[#6]-,:[#6]-,:[#6]=,:[#6]",
            "[#6]-,:[#7]:[#6&!H0]",
            "[#7]-,:[#6]:[#8]:[#6]",
            "[#8]=,:[#6]-,:[#6]:[#6]",
            "[#8]=,:[#6]-,:[#6]:[#7]",
            "[#6]-,:[#7]-,:[#6]:[#6]",
            "[#7]:[#7]-,:[#6&!H0]",
            "[#8]-,:[#6]:[#6]:[#7]",
            "[#8]-,:[#6]=,:[#6]-,:[#6]",
            "[#7]-,:[#6]:[#6]:[#7]",
            "[#6]-,:[#16]-,:[#6]:[#6]",
            "[Cl]-,:[#6]:[#6]-,:[#6]",
            "[#7]-,:[#6]=,:[#6&!H0]",
            "[Cl]-,:[#6]:[#6&!H0]",
            "[#7]:[#6]:[#7]-,:[#6]",
            "[Cl]-,:[#6]:[#6]-,:[#8]",
            "[#6]-,:[#6]:[#7]:[#6]",
            "[#6]-,:[#6]-,:[#16]-,:[#6]",
            "[#16]=,:[#6]-,:[#7]-,:[#6]",
            "[Br]-,:[#6]:[#6]-,:[#6]",
            "[#7&!H0]-,:[#7&!H0]",
            "[#16]=,:[#6]-,:[#7&!H0]",
            "[#6]-,:[#33]-[#8&!H0]",
            "[#16]:[#6]:[#6&!H0]",
            "[#8]-,:[#7]-,:[#6]-,:[#6]",
            "[#7]-,:[#7]-,:[#6]-,:[#6]",
            "[#6H,#6H2,#6H3]=,:[#6H,#6H2,#6H3]",
            "[#7]-,:[#7]-,:[#6]-,:[#7]",
            "[#8]=,:[#6]-,:[#7]-,:[#7]",
            "[#7]=,:[#6]-,:[#7]-,:[#6]",
            "[#6]=,:[#6]-,:[#6]:[#6]",
            "[#6]:[#7]-,:[#6&!H0]",
            "[#6]-,:[#7]-,:[#7&!H0]",
            "[#7]:[#6]:[#6]-,:[#6]",
            "[#6]-,:[#6]=,:[#6]-,:[#6]",
            "[#33]-,:[#6]:[#6&!H0]",
            "[Cl]-,:[#6]:[#6]-,:[Cl]",
            "[#6]:[#6]:[#7&!H0]",
            "[#7&!H0]-,:[#6&!H0]",
            "[Cl]-,:[#6]-,:[#6]-,:[Cl]",
            "[#7]:[#6]-,:[#6]:[#6]",
            "[#16]-,:[#6]:[#6]-,:[#6]",
            "[#16]-,:[#6]:[#6&!H0]",
            "[#16]-,:[#6]:[#6]-,:[#7]",
            "[#16]-,:[#6]:[#6]-,:[#8]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]",
            "[#8]=,:[#6]-,:[#6]-,:[#7]",
            "[#8]=,:[#6]-,:[#6]-,:[#8]",
            "[#7]=,:[#6]-,:[#6]-,:[#6]",
            "[#7]=,:[#6]-,:[#6&!H0]",
            "[#6]-,:[#7]-,:[#6&!H0]",
            "[#8]-,:[#6]:[#6]-,:[#6]",
            "[#8]-,:[#6]:[#6&!H0]",
            "[#8]-,:[#6]:[#6]-,:[#7]",
            "[#8]-,:[#6]:[#6]-,:[#8]",
            "[#7]-,:[#6]:[#6]-,:[#6]",
            "[#7]-,:[#6]:[#6&!H0]",
            "[#7]-,:[#6]:[#6]-,:[#7]",
            "[#8]-,:[#6]-,:[#6]:[#6]",
            "[#7]-,:[#6]-,:[#6]:[#6]",
            "[Cl]-,:[#6]-,:[#6]-,:[#6]",
            "[Cl]-,:[#6]-,:[#6]-,:[#8]",
            "[#6]:[#6]-,:[#6]:[#6]",
            "[#8]=,:[#6]-,:[#6]=,:[#6]",
            "[Br]-,:[#6]-,:[#6]-,:[#6]",
            "[#7]=,:[#6]-,:[#6]=,:[#6]",
            "[#6]=,:[#6]-,:[#6]-,:[#6]",
            "[#7]:[#6]-,:[#8&!H0]",
            "[#8]=,:[#7]-,:c:c",
            "[#8]-,:[#6]-,:[#7&!H0]",
            "[#7]-,:[#6]-,:[#7]-,:[#6]",
            "[Cl]-,:[#6]-,:[#6]=,:[#8]",
            "[Br]-,:[#6]-,:[#6]=,:[#8]",
            "[#8]-,:[#6]-,:[#8]-,:[#6]",
            "[#6]=,:[#6]-,:[#6]=,:[#6]",
            "[#6]:[#6]-,:[#8]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#7]",
            "[#8]-,:[#6]-,:[#6]-,:[#8]",
            "N#[#6]-,:[#6]-,:[#6]",
            "[#7]-,:[#6]-,:[#6]-,:[#7]",
            "[#6]:[#6]-,:[#6]-,:[#6]",
            "[#6&!H0]-,:[#8&!H0]",
            "n:c:n:c",
            "[#8]-,:[#6]-,:[#6]=,:[#6]",
            "[#8]-,:[#6]-,:[#6]:[#6]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]:[#6]-,:[#8]",
            "[#7]=,:[#6]-,:[#6]:[#6&!H0]",
            "c:c-,:[#7]-,:c:c",
            "[#6]-,:[#6]:[#6]-,:c:c",
            "[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#7]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#8]",
            "[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[Cl]-,:[#6]:[#6]-,:[#8]-,:[#6]",
            "c:c-,:[#6]=,:[#6]-,:[#6]",
            "[#6]-,:[#6]:[#6]-,:[#7]-,:[#6]",
            "[#6]-,:[#16]-,:[#6]-,:[#6]-,:[#6]",
            "[#7]-,:[#6]:[#6]-,:[#8&!H0]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]=,:[#8]",
            "[#6]-,:[#6]:[#6]-,:[#8]-,:[#6]",
            "[#6]-,:[#6]:[#6]-,:[#8&!H0]",
            "[Cl]-,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#7]-,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#7]-,:[#6]-,:[#6]-,:[#6]-,:[#7]",
            "[#6]-,:[#8]-,:[#6]-,:[#6]=,:[#6]",
            "c:c-,:[#6]-,:[#6]-,:[#6]",
            "[#7]=,:[#6]-,:[#7]-,:[#6]-,:[#6]",
            "[#8]=,:[#6]-,:[#6]-,:c:c",
            "[Cl]-,:[#6]:[#6]:[#6]-,:[#6]",
            "[#6H,#6H2,#6H3]-,:[#6]=,:[#6H,#6H2,#6H3]",
            "[#7]-,:[#6]:[#6]:[#6]-,:[#6]",
            "[#7]-,:[#6]:[#6]:[#6]-,:[#7]",
            "[#8]=,:[#6]-,:[#6]-,:[#7]-,:[#6]",
            "[#6]-,:c:c:[#6]-,:[#6]",
            "[#6]-,:[#8]-,:[#6]-,:[#6]:c",
            "[#8]=,:[#6]-,:[#6]-,:[#8]-,:[#6]",
            "[#8]-,:[#6]:[#6]-,:[#6]-,:[#6]",
            "[#7]-,:[#6]-,:[#6]-,:[#6]:c",
            "[#6]-,:[#6]-,:[#6]-,:[#6]:c",
            "[Cl]-,:[#6]-,:[#6]-,:[#7]-,:[#6]",
            "[#6]-,:[#8]-,:[#6]-,:[#8]-,:[#6]",
            "[#7]-,:[#6]-,:[#6]-,:[#7]-,:[#6]",
            "[#7]-,:[#6]-,:[#8]-,:[#6]-,:[#6]",
            "[#6]-,:[#7]-,:[#6]-,:[#6]-,:[#6]",
            "[#6]-,:[#6]-,:[#8]-,:[#6]-,:[#6]",
            "[#7]-,:[#6]-,:[#6]-,:[#8]-,:[#6]",
            "c:c:n:n:c",
            "[#6]-,:[#6]-,:[#6]-,:[#8&!H0]",
            "c:[#6]-,:[#6]-,:[#6]:c",
            "[#8]-,:[#6]-,:[#6]=,:[#6]-,:[#6]",
            "c:c-,:[#8]-,:[#6]-,:[#6]",
            "[#7]-,:[#6]:c:c:n",
            "[#8]=,:[#6]-,:[#8]-,:[#6]:c",
            "[#8]=,:[#6]-,:[#6]:[#6]-,:[#6]",
            "[#8]=,:[#6]-,:[#6]:[#6]-,:[#7]",
            "[#8]=,:[#6]-,:[#6]:[#6]-,:[#8]",
            "[#6]-,:[#8]-,:[#6]:[#6]-,:[#6]",
            "[#8]=,:[#33]-,:[#6]:c:c",
            "[#6]-,:[#7]-,:[#6]-,:[#6]:c",
            "[#16]-,:[#6]:c:c-,:[#7]",
            "[#8]-,:[#6]:[#6]-,:[#8]-,:[#6]",
            "[#8]-,:[#6]:[#6]-,:[#8&!H0]",
            "[#6]-,:[#6]-,:[#8]-,:[#6]:c",
            "[#7]-,:[#6]-,:[#6]:[#6]-,:[#6]",
            "[#6]-,:[#6]-,:[#6]:[#6]-,:[#6]",
            "[#7]-,:[#7]-,:[#6]-,:[#7&!H0]",
            "[#6]-,:[#7]-,:[#6]-,:[#7]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#7]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#8]",
            "[#6]=,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]=,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]=,:[#8]",
            "[#6&!H0]-,:[#6]-,:[#7&!H0]",
            "[#6]-,:[#6]=,:[#7]-,:[#7]-,:[#6]",
            "[#8]=,:[#6]-,:[#7]-,:[#6]-,:[#6]",
            "[#8]=,:[#6]-,:[#7]-,:[#6&!H0]",
            "[#8]=,:[#6]-,:[#7]-,:[#6]-,:[#7]",
            "[#8]=,:[#7]-,:[#6]:[#6]-,:[#7]",
            "[#8]=,:[#7]-,:c:c-,:[#8]",
            "[#8]=,:[#6]-,:[#7]-,:[#6]=,:[#8]",
            "[#8]-,:[#6]:[#6]:[#6]-,:[#6]",
            "[#8]-,:[#6]:[#6]:[#6]-,:[#7]",
            "[#8]-,:[#6]:[#6]:[#6]-,:[#8]",
            "[#7]-,:[#6]-,:[#7]-,:[#6]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]:c",
            "[#6]-,:[#6]-,:[#7]-,:[#6]-,:[#6]",
            "[#6]-,:[#7]-,:[#6]:[#6]-,:[#6]",
            "[#6]-,:[#6]-,:[#16]-,:[#6]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#7]-,:[#6]",
            "[#6]-,:[#6]=,:[#6]-,:[#6]-,:[#6]",
            "[#8]-,:[#6]-,:[#8]-,:[#6]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#8]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#8&!H0]",
            "[#6]-,:[#6]=,:[#6]-,:[#6]=,:[#6]",
            "[#7]-,:[#6]:[#6]-,:[#6]-,:[#6]",
            "[#6]=,:[#6]-,:[#6]-,:[#8]-,:[#6]",
            "[#6]=,:[#6]-,:[#6]-,:[#8&!H0]",
            "[#6]-,:[#6]:[#6]-,:[#6]-,:[#6]",
            "[Cl]-,:[#6]:[#6]-,:[#6]=,:[#8]",
            "[Br]-,:[#6]:c:c-,:[#6]",
            "[#8]=,:[#6]-,:[#6]=,:[#6]-,:[#6]",
            "[#8]=,:[#6]-,:[#6]=,:[#6&!H0]",
            "[#8]=,:[#6]-,:[#6]=,:[#6]-,:[#7]",
            "[#7]-,:[#6]-,:[#7]-,:[#6]:c",
            "[Br]-,:[#6]-,:[#6]-,:[#6]:c",
            "[#7]#[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#6]-,:[#6]=,:[#6]-,:[#6]:c",
            "[#6]-,:[#6]-,:[#6]=,:[#6]-,:[#6]",
            "[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#8]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#7]",
            "[#7]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#7]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#8]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]=,:[#8]",
            "[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#8]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#7]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#8]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]=,:[#8]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#7]",
            "[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#6])-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#6])-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#8]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#8])-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#7]-,:[#6]",
            "[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#7])-,:[#6]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#8])-,:[#6]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](=,:[#8])-,:[#6]",
            "[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#7])-,:[#6]",
            "[#6]-,:[#6](-,:[#6])-,:[#6]-,:[#6]",
            "[#6]-,:[#6](-,:[#6])-,:[#6]-,:[#6]-,:[#6]",
            "[#6]-,:[#6]-,:[#6](-,:[#6])-,:[#6]-,:[#6]",
            "[#6]-,:[#6](-,:[#6])(-,:[#6])-,:[#6]-,:[#6]",
            "[#6]-,:[#6](-,:[#6])-,:[#6](-,:[#6])-,:[#6]",
        ]
        return self._get_smarts_match_counts(mol, smarts_list)

    def _get_complex_smarts_patterns_counts(self, mol: Mol) -> list[int]:
        smarts_list = [
            "[#6]c1ccc([#6])cc1",
            "[#6]c1ccc([#8])cc1",
            "[#6]c1ccc([#16])cc1",
            "[#6]c1ccc([#7])cc1",
            "[#6]c1ccc(Cl)cc1",
            "[#6]c1ccc(Br)cc1",
            "[#8]c1ccc([#8])cc1",
            "[#8]c1ccc([#16])cc1",
            "[#8]c1ccc([#7])cc1",
            "[#8]c1ccc(Cl)cc1",
            "[#8]c1ccc(Br)cc1",
            "[#16]c1ccc([#16])cc1",
            "[#16]c1ccc([#7])cc1",
            "[#16]c1ccc(Cl)cc1",
            "[#16]c1ccc(Br)cc1",
            "[#7]c1ccc([#7])cc1",
            "[#7]c1ccc(Cl)cc1",
            "[#7]c1ccc(Br)cc1",
            "Clc1ccc(Cl)cc1",
            "Clc1ccc(Br)cc1",
            "Brc1ccc(Br)cc1",
            "[#6]c1cc([#6])ccc1",
            "[#6]c1cc([#8])ccc1",
            "[#6]c1cc([#16])ccc1",
            "[#6]c1cc([#7])ccc1",
            "[#6]c1cc(Cl)ccc1",
            "[#6]c1cc(Br)ccc1",
            "[#8]c1cc([#8])ccc1",
            "[#8]c1cc([#16])ccc1",
            "[#8]c1cc([#7])ccc1",
            "[#8]c1cc(Cl)ccc1",
            "[#8]c1cc(Br)ccc1",
            "[#16]c1cc([#16])ccc1",
            "[#16]c1cc([#7])ccc1",
            "[#16]c1cc(Cl)ccc1",
            "[#16]c1cc(Br)ccc1",
            "[#7]c1cc([#7])ccc1",
            "[#7]c1cc(Cl)ccc1",
            "[#7]c1cc(Br)ccc1",
            "Clc1cc(Cl)ccc1",
            "Clc1cc(Br)ccc1",
            "Brc1cc(Br)ccc1",
            "[#6]c1c([#6])cccc1",
            "[#6]c1c([#8])cccc1",
            "[#6]c1c([#16])cccc1",
            "[#6]c1c([#7])cccc1",
            "[#6]c1c(Cl)cccc1",
            "[#6]c1c(Br)cccc1",
            "[#8]c1c([#8])cccc1",
            "[#8]c1c([#16])cccc1",
            "[#8]c1c([#7])cccc1",
            "[#8]c1c(Cl)cccc1",
            "[#8]c1c(Br)cccc1",
            "[#16]c1c([#16])cccc1",
            "[#16]c1c([#7])cccc1",
            "[#16]c1c(Cl)cccc1",
            "[#16]c1c(Br)cccc1",
            "[#7]c1c([#7])cccc1",
            "[#7]c1c(Cl)cccc1",
            "[#7]c1c(Br)cccc1",
            "Clc1c(Cl)cccc1",
            "Clc1c(Br)cccc1",
            "Brc1c(Br)cccc1",
            "[#6][#6]1[#6][#6][#6]([#6])[#6][#6]1",
            "[#6][#6]1[#6][#6][#6]([#8])[#6][#6]1",
            "[#6][#6]1[#6][#6][#6]([#16])[#6][#6]1",
            "[#6][#6]1[#6][#6][#6]([#7])[#6][#6]1",
            "[#6][#6]1[#6][#6][#6](Cl)[#6][#6]1",
            "[#6][#6]1[#6][#6][#6](Br)[#6][#6]1",
            "[#8][#6]1[#6][#6][#6]([#8])[#6][#6]1",
            "[#8][#6]1[#6][#6][#6]([#16])[#6][#6]1",
            "[#8][#6]1[#6][#6][#6]([#7])[#6][#6]1",
            "[#8][#6]1[#6][#6][#6](Cl)[#6][#6]1",
            "[#8][#6]1[#6][#6][#6](Br)[#6][#6]1",
            "[#16][#6]1[#6][#6][#6]([#16])[#6][#6]1",
            "[#16][#6]1[#6][#6][#6]([#7])[#6][#6]1",
            "[#16][#6]1[#6][#6][#6](Cl)[#6][#6]1",
            "[#16][#6]1[#6][#6][#6](Br)[#6][#6]1",
            "[#7][#6]1[#6][#6][#6]([#7])[#6][#6]1",
            "[#7][#6]1[#6][#6][#6](Cl)[#6][#6]1",
            "[#7][#6]1[#6][#6][#6](Br)[#6][#6]1",
            "Cl[#6]1[#6][#6][#6](Cl)[#6][#6]1",
            "Cl[#6]1[#6][#6][#6](Br)[#6][#6]1",
            "Br[#6]1[#6][#6][#6](Br)[#6][#6]1",
            "[#6][#6]1[#6][#6]([#6])[#6][#6][#6]1",
            "[#6][#6]1[#6][#6]([#8])[#6][#6][#6]1",
            "[#6][#6]1[#6][#6]([#16])[#6][#6][#6]1",
            "[#6][#6]1[#6][#6]([#7])[#6][#6][#6]1",
            "[#6][#6]1[#6][#6](Cl)[#6][#6][#6]1",
            "[#6][#6]1[#6][#6](Br)[#6][#6][#6]1",
            "[#8][#6]1[#6][#6]([#8])[#6][#6][#6]1",
            "[#8][#6]1[#6][#6]([#16])[#6][#6][#6]1",
            "[#8][#6]1[#6][#6]([#7])[#6][#6][#6]1",
            "[#8][#6]1[#6][#6](Cl)[#6][#6][#6]1",
            "[#8][#6]1[#6][#6](Br)[#6][#6][#6]1",
            "[#16][#6]1[#6][#6]([#16])[#6][#6][#6]1",
            "[#16][#6]1[#6][#6]([#7])[#6][#6][#6]1",
            "[#16][#6]1[#6][#6](Cl)[#6][#6][#6]1",
            "[#16][#6]1[#6][#6](Br)[#6][#6][#6]1",
            "[#7][#6]1[#6][#6]([#7])[#6][#6][#6]1",
            "[#7][#6]1[#6][#6](Cl)[#6][#6][#6]1",
            "[#7][#6]1[#6][#6](Br)[#6][#6][#6]1",
            "Cl[#6]1[#6][#6](Cl)[#6][#6][#6]1",
            "Cl[#6]1[#6][#6](Br)[#6][#6][#6]1",
            "Br[#6]1[#6][#6](Br)[#6][#6][#6]1",
            "[#6][#6]1[#6]([#6])[#6][#6][#6][#6]1",
            "[#6][#6]1[#6]([#8])[#6][#6][#6][#6]1",
            "[#6][#6]1[#6]([#16])[#6][#6][#6][#6]1",
            "[#6][#6]1[#6]([#7])[#6][#6][#6][#6]1",
            "[#6][#6]1[#6](Cl)[#6][#6][#6][#6]1",
            "[#6][#6]1[#6](Br)[#6][#6][#6][#6]1",
            "[#8][#6]1[#6]([#8])[#6][#6][#6][#6]1",
            "[#8][#6]1[#6]([#16])[#6][#6][#6][#6]1",
            "[#8][#6]1[#6]([#7])[#6][#6][#6][#6]1",
            "[#8][#6]1[#6](Cl)[#6][#6][#6][#6]1",
            "[#8][#6]1[#6](Br)[#6][#6][#6][#6]1",
            "[#16][#6]1[#6]([#16])[#6][#6][#6][#6]1",
            "[#16][#6]1[#6]([#7])[#6][#6][#6][#6]1",
            "[#16][#6]1[#6](Cl)[#6][#6][#6][#6]1",
            "[#16][#6]1[#6](Br)[#6][#6][#6][#6]1",
            "[#7][#6]1[#6]([#7])[#6][#6][#6][#6]1",
            "[#7][#6]1[#6](Cl)[#6][#6][#6][#6]1",
            "[#7][#6]1[#6](Br)[#6][#6][#6][#6]1",
            "Cl[#6]1[#6](Cl)[#6][#6][#6][#6]1",
            "Cl[#6]1[#6](Br)[#6][#6][#6][#6]1",
            "Br[#6]1[#6](Br)[#6][#6][#6][#6]1",
            "[#6][#6]1[#6][#6]([#6])[#6][#6]1",
            "[#6][#6]1[#6][#6]([#8])[#6][#6]1",
            "[#6][#6]1[#6][#6]([#16])[#6][#6]1",
            "[#6][#6]1[#6][#6]([#7])[#6][#6]1",
            "[#6][#6]1[#6][#6](Cl)[#6][#6]1",
            "[#6][#6]1[#6][#6](Br)[#6][#6]1",
            "[#8][#6]1[#6][#6]([#8])[#6][#6]1",
            "[#8][#6]1[#6][#6]([#16])[#6][#6]1",
            "[#8][#6]1[#6][#6]([#7])[#6][#6]1",
            "[#8][#6]1[#6][#6](Cl)[#6][#6]1",
            "[#8][#6]1[#6][#6](Br)[#6][#6]1",
            "[#16][#6]1[#6][#6]([#16])[#6][#6]1",
            "[#16][#6]1[#6][#6]([#7])[#6][#6]1",
            "[#16][#6]1[#6][#6](Cl)[#6][#6]1",
            "[#16][#6]1[#6][#6](Br)[#6][#6]1",
            "[#7][#6]1[#6][#6]([#7])[#6][#6]1",
            "[#7][#6]1[#6][#6](Cl)[#6][#6]1",
            "[#7][#6]1[#6][#6](Br)[#6][#6]1",
            "Cl[#6]1[#6][#6](Cl)[#6][#6]1",
            "Cl[#6]1[#6][#6](Br)[#6][#6]1",
            "Br[#6]1[#6][#6](Br)[#6][#6]1",
            "[#6][#6]1[#6]([#6])[#6][#6][#6]1",
            "[#6][#6]1[#6]([#8])[#6][#6][#6]1",
            "[#6][#6]1[#6]([#16])[#6][#6][#6]1",
            "[#6][#6]1[#6]([#7])[#6][#6][#6]1",
            "[#6][#6]1[#6](Cl)[#6][#6][#6]1",
            "[#6][#6]1[#6](Br)[#6][#6][#6]1",
            "[#8][#6]1[#6]([#8])[#6][#6][#6]1",
            "[#8][#6]1[#6]([#16])[#6][#6][#6]1",
            "[#8][#6]1[#6]([#7])[#6][#6][#6]1",
            "[#8][#6]1[#6](Cl)[#6][#6][#6]1",
            "[#8][#6]1[#6](Br)[#6][#6][#6]1",
            "[#16][#6]1[#6]([#16])[#6][#6][#6]1",
            "[#16][#6]1[#6]([#7])[#6][#6][#6]1",
            "[#16][#6]1[#6](Cl)[#6][#6][#6]1",
            "[#16][#6]1[#6](Br)[#6][#6][#6]1",
            "[#7][#6]1[#6]([#7])[#6][#6][#6]1",
            "[#7][#6]1[#6](Cl)[#6][#6]1",
            "[#7][#6]1[#6](Br)[#6][#6][#6]1",
            "Cl[#6]1[#6](Cl)[#6][#6][#6]1",
            "Cl[#6]1[#6](Br)[#6][#6][#6]1",
            "Br[#6]1[#6](Br)[#6][#6][#6]1",
        ]
        return self._get_smarts_match_counts(mol, smarts_list)

    def _get_smarts_match_counts(self, mol: Mol, smarts_list: list[str]) -> list[int]:
        from rdkit.Chem import MolFromSmarts

        return [
            len(mol.GetSubstructMatches(MolFromSmarts(smarts)))
            for smarts in smarts_list
        ]
