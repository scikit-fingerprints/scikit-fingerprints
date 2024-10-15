from typing import Union

from rdkit.Chem import AssignStereochemistry, GetFormalCharge, Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import (
    CalcNumAtomStereoCenters,
    CalcNumHBA,
    CalcNumHBD,
    CalcNumHeteroatoms,
    CalcNumRings,
    CalcNumRotatableBonds,
    CalcTPSA,
)

from skfp.bases.base_filter import BaseFilter

from .utils import (
    get_max_ring_size,
    get_non_carbon_to_carbon_ratio,
    get_num_carbon_atoms,
    get_num_charged_functional_groups,
    get_num_rigid_bonds,
)


class FAF4LeadlikeFilter(BaseFilter):
    """
    FAFDrugs4 Lead-Like Soft filter.

    Designed as a part of FAFDrugs4 software [1]_ [2]_. Based on literature describing
    physico-chemical properties of lead drugs. Designed to keep starting point molecules
    that can be further optimized, i.e. relatively small, with low logP, and that can
    be "decorated" further to increase affinity and or selectivity without becoming
    very ADMET unfriendly).

    Basically a more restrictive variant of FAFDrugs4 Drug-Like Soft filter.

    Molecule must fulfill conditions:

    - molecular weight in range ``[150, 400]``
    - logP in range ``[-3, 4]``
    - HBA <= 7
    - HBD <= 4
    - TPSA <= 160
    - number of rotatable bonds <= 9
    - number of rigid bonds <= 30
    - number of rings <= 4
    - max ring size <= 18
    - number of carbons in range ``[3, 35]``
    - number of heteroatoms in range ``[1, 15]``
    - non-carbons to carbons ratio in range ``[0.1, 1.1]``
    - number of charged functional groups <= 4
    - total formal charge in range ``[-4, 4]``
    - number of stereocenters <= 2

    Note that the FAF4Drugs uses ChemAxon for determining functional groups. We use
    their publicly available CXSMARTS list of functional groups [3]_. Phosphine and
    sulfoxide patterns could not be parsed by RDKit, so we manually fixed them.

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
        processors. See Scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int, default=0
        Controls the verbosity when filtering molecules.

    References
    -----------
    .. [1] `Details of physico-chemical property filters available in FAF-Drugs4
        <https://fafdrugs4.rpbs.univ-paris-diderot.fr/filters.html>`_

    .. [2] `D. Lagorce et al.
        "FAF-Drugs4: free ADME-tox filtering computations for chemical biology and
        early stages drug discovery"
        Bioinformatics, 33(22), 2017, 3658–3660
        <https://doi.org/10.1093/bioinformatics/btx491>`_

    .. [3] `ChemAxon documentation: Predefined Functional Groups and Named Molecule Groups
        <https://docs.chemaxon.com/display/docs/attachments/attachments_1829721_1_functionalgroups.cxsmi>`_

    Examples
    ----------
    >>> from skfp.filters import FAF4LeadlikeFilter
    >>> smiles = ["C", "CC(=O)Nc1ccc(O)cc1"]
    >>> filt = FAF4LeadlikeFilter()
    >>> filt
    FAF4LeadlikeFilter()

    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['CC(=O)Nc1ccc(O)cc1']
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
        # explicit "copy" via SMILES due to RDKit bug
        # https://github.com/rdkit/rdkit/issues/7917
        from rdkit.Chem import MolFromSmiles, MolToSmiles

        mol = MolFromSmiles(MolToSmiles(mol))
        AssignStereochemistry(mol)

        rules = [
            150 <= MolWt(mol) <= 400,
            -3 <= MolLogP(mol) <= 4,
            CalcNumHBA(mol) <= 7,
            CalcNumHBD(mol) <= 4,
            CalcTPSA(mol) <= 160,
            CalcNumRotatableBonds(mol) <= 9,
            get_num_rigid_bonds(mol) <= 30,
            CalcNumRings(mol) <= 4,
            get_max_ring_size(mol) <= 18,
            3 <= get_num_carbon_atoms(mol) <= 35,
            1 <= CalcNumHeteroatoms(mol) <= 15,
            0.1 <= get_non_carbon_to_carbon_ratio(mol) <= 1.1,
            get_num_charged_functional_groups(mol) <= 4,
            -4 <= GetFormalCharge(mol) <= 4,
            CalcNumAtomStereoCenters(mol) <= 2,
        ]
        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
