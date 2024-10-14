from typing import Union

from rdkit.Chem import Crippen, Mol, rdMolDescriptors, rdmolops

from skfp.bases.base_filter import BaseFilter

from .utils import (
    get_hc_ratio,
    get_max_ring_size,
    get_num_carbon_atoms,
    get_num_charged_functional_groups,
    get_num_rigid_bonds,
)


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
    - number of rigid bonds <= 30
    - number of rings in range <= 6
    - max size of A ring <= 18
    - number of carbon atoms in range ``[3, 35]``
    - number of heteroatoms in range ``[1, 15]``
    - number of heavy atoms in range ``[10, 50]``
    - HC ratio in range ``[0.1, 1.1]``
    - charge in range ``[-4, 4]``
    - number of charged functional groups <= 4

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
        TODO
        <https://fafdrugs4.rpbs.univ-paris-diderot.fr/filters.html>`_

    Examples
    ----------
    >>> from skfp.preprocessing import RuleOfDrugLikeSoft
    >>> smiles = [TODO]
    >>> filt = RuleOfDrugLikeSoft()
    >>> filt
    RuleOfDrugLikeSoft()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    [TODO]
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
            get_num_rigid_bonds(mol) <= 30,
            rdMolDescriptors.CalcNumRings(mol) <= 6,
            get_max_ring_size(mol) <= 18,
            3 <= get_num_carbon_atoms(mol) <= 35,
            1 <= rdMolDescriptors.CalcNumHeteroatoms(mol) <= 15,
            0.1 <= get_hc_ratio(mol) <= 1.1,
            -4 <= rdmolops.GetFormalCharge(mol) <= 4,
            get_num_charged_functional_groups(mol) <= 4,
        ]

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
