from typing import Union

from rdkit.Chem import GetFormalCharge, Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import (
    CalcNumHBA,
    CalcNumHBD,
    CalcNumHeteroatoms,
    CalcNumRotatableBonds,
    CalcTPSA,
)

from skfp.bases.base_filter import BaseFilter

from .utils import (
    get_max_num_fused_aromatic_rings,
    get_max_ring_size,
    get_num_aromatic_rings,
    get_num_carbon_atoms,
    get_num_charged_atoms,
    get_num_heavy_metals,
    get_num_rigid_bonds,
)


class ValenceDiscoveryFilter(BaseFilter):
    """
    Valence Discovery © generative design filter.

    Set of rules proprietary to Valence Discovery ©, curated to prioritize molecules
    for generative models. Based on definitions from MedChem [1]_.

    Molecule must fulfill conditions:

    - molecular weight in range ``[200, 600]``
    - logP in range ``[-3, 6]``
    - HBA <= 12
    - HBD <= 7
    - TPSA in range ``[40, 180]``
    - number of rotatable bonds <= 15
    - number of rigid bonds <= 30
    - number of aromatic rings <= 5
    - number of aromatic rings fused together <= 2
    - max ring size <= 18
    - number of heavy atoms < 70
    - number of heavy metals < 1
    - number of carbons in range ``[3, 40]``
    - number of heteroatoms in range ``[1, 15]``
    - total formal charge in range ``[-2, 2]``
    - number of charged atoms <= 2

    Heavy atoms are defined as metals other than ["Li", "Be", "K", "Na", "Ca", "Mg"].

    Parameters
    ----------
    allow_one_violation : bool, default=False
        Whether to allow violating one of the rules for a molecule. This makes the
        filter less restrictive.

    return_indicators : bool, default=False
        Whether to return a binary vector with indicators which molecules pass the
        filter, instead of list of molecules.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform_x_y` and
        :meth:`transform` are parallelized over the input molecules. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int, default=0
        Controls the verbosity when filtering molecules.

    References
    ----------
    .. [1] `MedChem documentation - rule_of_generative_design
        <https://medchem-docs.datamol.io/stable/api/medchem.rules.html#medchem.rules.basic_rules.rule_of_generative_design>`_

    Examples
    --------
    >>> from skfp.filters import ValenceDiscoveryFilter
    >>> smiles = ["C", "ClC1=CC2=C(N=C(NC)C[N+]([O-])=C2C3=CC=CC=C3)C=C1"]
    >>> filt = ValenceDiscoveryFilter()
    >>> filt
    ValenceDiscoveryFilter()

    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['ClC1=CC2=C(N=C(NC)C[N+]([O-])=C2C3=CC=CC=C3)C=C1']
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
            200 <= MolWt(mol) <= 600,
            -3 <= MolLogP(mol) <= 6,
            CalcNumHBA(mol) <= 12,
            CalcNumHBD(mol) <= 7,
            40 <= CalcTPSA(mol) <= 180,
            CalcNumRotatableBonds(mol) <= 15,
            get_num_rigid_bonds(mol) <= 30,
            get_num_aromatic_rings(mol) <= 5,
            get_max_num_fused_aromatic_rings(mol) <= 2,
            get_max_ring_size(mol) <= 18,
            mol.GetNumHeavyAtoms() < 70,
            get_num_heavy_metals(mol) < 1,
            3 <= get_num_carbon_atoms(mol) <= 40,
            1 <= CalcNumHeteroatoms(mol) <= 15,
            -2 <= GetFormalCharge(mol) <= 2,
            get_num_charged_atoms(mol) <= 2,
        ]
        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
