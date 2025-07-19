import numpy as np
from rdkit.Chem import GetFormalCharge, Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import (
    CalcExactMolWt,
    CalcNumHBA,
    CalcNumHBD,
    CalcNumHeavyAtoms,
    CalcNumRotatableBonds,
)

from skfp.bases.base_filter import BaseFilter


class REOSFilter(BaseFilter):
    """
    REOS filter.

    REOS (Rapid Elimination Of Swill) is designed to filter out molecules with
    undesirable properties for drug discovery [1]_.

    Molecule must fulfill conditions:

    - molecular weight in range [200, 500]
    - logP in range [-5, 5]
    - HBA <= 10
    - HBD <= 5
    - charge in range [-2, 2]
    - number of rotatable bonds <= 8
    - number of heavy atoms in range [15, 50]

    Parameters
    ----------
    allow_one_violation : bool, default=False
        Whether to allow violating one of the rules for a molecule. This makes the
        filter less restrictive.

    return_type : {"mol", "indicators", "condition_indicators"}, default="mol"
        What values to return as the filtering result. `"mol"` returns list of
        molecules passing the filter. `"indicators"` returns a binary vector with
        indicators which molecules pass the filter. `"condition_indicators"` returns
        a Pandas DataFrame with molecules in rows, filter conditions in columns, and
        0/1 indicators whether a given condition was fulfilled by a given molecule.

    return_indicators : bool, default=False
        Whether to return a binary vector with indicators which molecules pass the
        filter, instead of list of molecules.

        .. deprecated:: 1.17
            ``return_indicators`` is deprecated and will be removed in version 2.0.
            Use ``return_type`` instead. If ``return_indicators`` is set to ``True``,
            it will take precedence over ``return_type``.

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
        return_type: str = "mol",
        return_indicators: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            allow_one_violation=allow_one_violation,
            return_type=return_type,
            return_indicators=return_indicators,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self._condition_names = [
            "200 <= MolWeight <= 500",
            "-5 <= logP <= 5",
            "HBA <= 10",
            "HBD <= 5",
            "-2 <= formal charge <= 2",
            "rotatable bonds <= 8",
            "15 <= heavy atoms <= 50",
        ]

    def _apply_mol_filter(self, mol: Mol) -> bool | np.ndarray:
        rules = [
            200 <= CalcExactMolWt(mol) <= 500,
            -5 <= MolLogP(mol) <= 5,
            CalcNumHBA(mol) <= 10,
            CalcNumHBD(mol) <= 5,
            -2 <= GetFormalCharge(mol) <= 2,
            CalcNumRotatableBonds(mol) <= 8,
            15 <= CalcNumHeavyAtoms(mol) <= 50,
        ]

        if self.return_type == "condition_indicators":
            return np.array(rules, dtype=bool)

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
