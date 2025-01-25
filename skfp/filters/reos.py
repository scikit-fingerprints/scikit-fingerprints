from typing import Union

from rdkit.Chem import Crippen, Mol, rdMolDescriptors, rdmolops

from skfp.bases.base_filter import BaseFilter


class REOSFilter(BaseFilter):
    """
    REOS filter.

    REOS (Rapid Elimination Of Swill) is designed to filter out molecules with
    undesirable properties for drug discovery [1]_.

    Molecule must fulfill conditions:

    - molecular weight in range ``[200, 500]``
    - logP in range ``[-5, 5]``
    - HBA in range ``[0, 5]``
    - HBD in range ``[0, 10]``
    - charge in range ``[-2, 2]``
    - number of rotatable bonds in range ``[0, 8]``
    - number of heavy atoms in range ``[15, 50]``

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

    verbose : int or dict, default=0
        Controls the verbosity when filtering molecules.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    References
    ----------
    .. [1] `Walters, W. P., Namchuk, M.
        "Designing screens: how to make your hits a hit"
        Nat Rev Drug Discov. 2003 Apr;2(4):259-66
        <https://pubmed.ncbi.nlm.nih.gov/12669025/>`_

    Examples
    --------
    >>> from skfp.filters import REOSFilter
    >>> smiles = ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  "CC(=O)c1c(C(C)=O)c(C)n(CCCCn2c(C)c(C(C)=O)c(C(C)=O)c2C)c1C"]
    >>> filt = REOSFilter()
    >>> filt
    REOSFilter()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['CC(C)CC1=CC=C(C=C1)C(C)C(=O)O']
    """

    def __init__(
        self,
        allow_one_violation: bool = False,
        return_indicators: bool = False,
        n_jobs: Union[int, None] = None,
        batch_size: Union[int, None] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            allow_one_violation, return_indicators, n_jobs, batch_size, verbose
        )

    def _apply_mol_filter(self, mol: Mol) -> bool:
        rules = [
            200 <= rdMolDescriptors.CalcExactMolWt(mol) <= 500,
            -5 <= Crippen.MolLogP(mol) <= 5,
            0 <= rdMolDescriptors.CalcNumHBA(mol) <= 10,
            0 <= rdMolDescriptors.CalcNumHBD(mol) <= 5,
            -2 <= rdmolops.GetFormalCharge(mol) <= 2,
            0 <= rdMolDescriptors.CalcNumRotatableBonds(mol) <= 8,
            15 <= rdMolDescriptors.CalcNumHeavyAtoms(mol) <= 50,
        ]

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
