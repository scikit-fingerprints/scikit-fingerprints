from typing import Optional, Union

from rdkit.Chem import Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumRings

from skfp.bases.base_filter import BaseFilter


class RuleOfFourFilter(BaseFilter):
    """
    Rule of four (Ro4).

    Rule designed to look for molecules used as PPI (protein-protein inhibitor).
    Described in [1]_.

    Molecule must fulfill conditions:

    - molecular weight >= 400 daltons
    - HBA >= 4
    - logP >=4
    - number of rings >= 4

    Parameters
    ------------
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

    verbose : int or dict, default=0
        Controls the verbosity when filtering molecules.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    References
    -----------
    .. [1] `Morelli, X., Bourgeas, R., & Roche, P.
        "Chemical and structural lessons from recent successes in protein–protein interaction inhibition (2P2I)."
        Current Opinion in Chemical Biology, 15(4), 475–481.
        <https://doi.org/10.1016/j.cbpa.2011.05.024>`_

    Examples
    ---------
    >>> from skfp.filters import RuleOfFourFilter
    >>> smiles = ['c1ccc2oc(-c3ccc(Nc4nc(N5CCCCC5)nc(N5CCOCC5)n4)cc3)nc2c1', \
    'c1nc(N2CCOCC2)c2sc3nc(N4CCOCC4)c4c(c3c2n1)CCCC4']
    >>> filt = RuleOfFourFilter()
    >>> filt
    RuleOfFourFilter()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['c1ccc2oc(-c3ccc(Nc4nc(N5CCCCC5)nc(N5CCOCC5)n4)cc3)nc2c1']
    """

    def __init__(
        self,
        allow_one_violation: bool = False,
        return_indicators: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            allow_one_violation=allow_one_violation,
            return_indicators=return_indicators,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def _apply_mol_filter(self, mol: Mol) -> bool:
        rules = [
            MolWt(mol) >= 400,
            MolLogP(mol) >= 4,
            CalcNumHBA(mol) >= 4,
            CalcNumRings(mol) >= 4,
        ]

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
