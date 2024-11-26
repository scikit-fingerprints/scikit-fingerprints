from typing import Optional, Union

from rdkit.Chem import Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA

from skfp.bases.base_filter import BaseFilter


class PfizerFilter(BaseFilter):
    """
    Pfizer 3/75 rule.

    Based on observation that compounds exhibiting low partition coefficient (clogP) and
    high topological polar surface area (TPSA) are roughly 2.5 times more likely to be
    free of toxicity issues in the tested conditions [1]_ [2]_.

    Molecule must fulfill conditions:

    - logP <= 3
    - TPSA >= 75

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

    verbose : int or dict, default=0
        Controls the verbosity when filtering molecules.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    References
    -----------
    .. [1] `Hughes, J. D. et al.
        "Physiochemical drug properties associated with in vivo toxicological outcomes."
        Bioorganic & Medicinal Chemistry Letters, 18(17), 4872–4875.
        <https://doi.org/10.1016/j.bmcl.2008.07.071>`_

    .. [2] `Price, D. A., Blagg, J., Jones, L., Greene, N., & Wager, T.
        "Physicochemical drug properties associated with in vivo toxicological outcomes: a review."
        Expert Opinion on Drug Metabolism & Toxicology, 5(8), 921–931.
        <https://doi.org/10.1517/17425250903042318>`_

    Examples
    ----------
    >>> from skfp.filters import PfizerFilter
    >>> smiles = ["CS(=O)(=O)NCc1nnc(SCc2ccccc2C(F)(F)F)o1", "COC(=O)c1ccccc1NC(=O)CSc1nc(O)c(-c2ccccc2)c(=O)[nH]1",\
    "Cc1ccccc1-n1c(Cc2cccs2)n[nH]c1=S"]
    >>> filt = PfizerFilter()
    >>> filt
    PfizerFilter()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['CS(=O)(=O)NCc1nnc(SCc2ccccc2C(F)(F)F)o1', 'COC(=O)c1ccccc1NC(=O)CSc1nc(O)c(-c2ccccc2)c(=O)[nH]1']
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
            MolLogP(mol) <= 3,
            CalcTPSA(mol) >= 75,
        ]
        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
