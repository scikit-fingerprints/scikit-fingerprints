from typing import Union

from rdkit.Chem import Crippen, Mol, rdMolDescriptors, rdmolops

from skfp.bases.base_filter import BaseFilter


class RuleOfDrugLikeSoft(BaseFilter):
    """
    Rule of DrugLike Soft.

    Compute the DrugLike Soft rule [1]_.

    Molecule must fulfill conditions:

    - molecular weight in range ``[100, 600]``
    - logP in range ``[-3, 6]``
    - HBD <= 7
    - HBA <= 12
    - TPSA <= 180
    - number of rotatable bonds <= 11
    - RIGBONDS <= 30
    - number of rings in range <= 6
    - max size of A ring <= 18
    - number of carbon atoms in range ``[3, 35]``
    - number of heteroatoms in range ``[1, 15]``
    - number of heavy atoms in range ``[10, 50]``
    - HC ratio in range ``[0.1, 1.1]``
    - charge in range ``[-4, 4]``
    - number of atoms carrying a charge <= 4

    Parameters
    ----------
    allow_one_violation : bool, default=False
        Whether to allow violating one of the rules for a molecule. This makes the
        filter less restrictive.

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

    References
    -----------
    .. [1] `
        <https://fafdrugs4.rpbs.univ-paris-diderot.fr/filters.html>`_

    Examples
    ----------
    >>> from skfp.preprocessing import RuleOfDrugLikeSoft
    >>> smiles = ["CCO", "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O"]
    >>> filt = RuleOfDrugLikeSoft()
    >>> filt
    RuleOfDrugLikeSoft()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O']
    """

    def __init__(
        self,
        allow_one_violation: bool = False,
        return_indicators: bool = False,
        n_jobs: Union[int, None] = None,
        batch_size: Union[int, None] = None,
        verbose: int = 0,
    ):
        super().__init__(
            allow_one_violation, return_indicators, n_jobs, batch_size, verbose
        )

    def _apply_mol_filter(self, mol: Mol) -> bool:
        rules = [
            100 <= rdMolDescriptors.CalcExactMolWt(mol) <= 600,
            -3 <= Crippen.MolLogP(mol) <= 6,
            rdMolDescriptors.CalcNumHBD(mol) <= 7,
            rdMolDescriptors.CalcNumHBA(mol) <= 12,
            rdMolDescriptors.CalcTPSA(mol) <= 180,
            rdMolDescriptors.CalcNumRotatableBonds(mol) <= 11,
            self._get_rigbonds(mol) <= 30,
            rdMolDescriptors.CalcNumRings(mol) <= 6,
            self._get_max_ring_size(mol) <= 18,
            3 <= self._get_number_of_carbons(mol) <= 35,
            1 <= rdMolDescriptors.CalcNumHeteroatoms(mol) <= 15,
            0.1 <= self._get_hc_ratio(mol) <= 1.1,
            -4 <= rdmolops.GetFormalCharge(mol) <= 4,
            # TODO: N_ATOM_CHARGE <= 4
        ]

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)

    def _get_max_ring_size(self, mol: Mol) -> int:
        rings = mol.GetRingInfo().AtomRings()

        return max(len(ring) for ring in rings) if rings else 0

    def _get_number_of_carbons(self, mol: Mol) -> int:
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")

    def _get_hc_ratio(self, mol: Mol) -> float:
        num_carbons: int = self._get_number_of_carbons(mol)
        num_hydrogens = sum(
            atom.GetNumImplicitHs() + atom.GetExplicitValence()
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "C"
        )

        if num_carbons > 0:
            return num_hydrogens / num_carbons
        else:
            return 0.0

    def _get_rigbonds(self, mol: Mol) -> int:
        total_bonds: int = mol.GetNumBonds()
        rotatable_bonds: int = rdMolDescriptors.CalcNumRotatableBonds(mol)

        return total_bonds - rotatable_bonds


if __name__ == "__main__":
    smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    filter = RuleOfDrugLikeSoft()

    from rdkit.Chem import MolFromSmiles

    mol = MolFromSmiles(smiles)
    rule_set = [
        rdMolDescriptors.CalcExactMolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        filter._get_rigbonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        filter._get_max_ring_size(mol),
        filter._get_number_of_carbons(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        filter._get_hc_ratio(mol),
        rdmolops.GetFormalCharge(mol),
        # TODO: N_ATOM_CHARGE <= 4
    ]

    rule_bools = [
        100 <= rdMolDescriptors.CalcExactMolWt(mol) <= 600,
        -3 <= Crippen.MolLogP(mol) <= 6,
        rdMolDescriptors.CalcNumHBD(mol) <= 7,
        rdMolDescriptors.CalcNumHBA(mol) <= 12,
        rdMolDescriptors.CalcTPSA(mol) <= 180,
        rdMolDescriptors.CalcNumRotatableBonds(mol) <= 11,
        filter._get_rigbonds(mol) <= 30,
        rdMolDescriptors.CalcNumRings(mol) <= 6,
        filter._get_max_ring_size(mol) <= 18,
        3 <= filter._get_number_of_carbons(mol) <= 35,
        1 <= rdMolDescriptors.CalcNumHeteroatoms(mol) <= 15,
        0.1 <= filter._get_hc_ratio(mol) <= 1.1,
        -4 <= rdmolops.GetFormalCharge(mol) <= 4,
        # TODO: N_ATOM_CHARGE <= 4
    ]

    print(rule_set)
    print(rule_bools)
