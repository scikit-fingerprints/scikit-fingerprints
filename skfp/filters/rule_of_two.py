import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD

from skfp.bases.base_filter import BaseFilter


class RuleOfTwoFilter(BaseFilter):
    """
    Rule of two (Ro2).

    Designed for finding reagents for building block design [1]_.

    Molecule must fulfill conditions:

    - molecular weight <= 200
    - HBA <= 4
    - HBD <= 2
    - logP <= 2

    Parameters
    ----------
    allow_one_violation : bool, default=False
        Whether to allow violating one of the rules for a molecule. This makes the
        filter less restrictive.

    return_type : {"mol", "indicators", "condition_indicators"}, default="mol"
        What values to return as the filtering result. "mol" returns list of
        molecules passing the filter. "indicators" returns a binary vector with
        indicators which molecules pass the filter. "condition_indicators" returns
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
    .. [1] `Goldberg, F. W., Kettle, J. G., Kogej, T., Perry, M. W. D., & Tomkinson, N. P.
        "Designing novel building blocks is an overlooked strategy to improve compound quality."
        Drug Discovery Today, 20(1), 11-17.
        <https://doi.org/10.1016/j.drudis.2014.09.023>`_

    Examples
    --------
    >>> from skfp.filters import RuleOfTwoFilter
    >>> smiles = ['C=CCc1c(C)[nH]c(N)nc1=O', 'C=CCNC(=O)c1ccncc1', 'C=CCC1C=C(C)CC(CC=C)N1']
    >>> filt = RuleOfTwoFilter()
    >>> filt
    RuleOfTwoFilter()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['C=CCc1c(C)[nH]c(N)nc1=O', 'C=CCNC(=O)c1ccncc1']
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
        condition_names = [
            "MolWeight <= 200",
            "HBA <= 4",
            "HBD <= 2",
            "logP <= 2",
        ]
        super().__init__(
            condition_names=condition_names,
            allow_one_violation=allow_one_violation,
            return_type=return_type,
            return_indicators=return_indicators,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def _apply_mol_filter(self, mol: Mol) -> bool | np.ndarray:
        rules = [
            MolWt(mol) <= 200,
            CalcNumHBA(mol) <= 4,
            CalcNumHBD(mol) <= 2,
            MolLogP(mol) <= 2,
        ]

        if self.return_type == "condition_indicators":
            return np.array(rules, dtype=bool)

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
