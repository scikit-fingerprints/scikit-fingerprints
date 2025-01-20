from typing import Optional, Union

from rdkit.Chem import Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD, CalcNumRotatableBonds

from skfp.bases.base_filter import BaseFilter


class TiceInsecticidesFilter(BaseFilter):
    r"""
    Tice rule for insecticides.

    Rule established based on statistical analysis of insecticides molecules [1]_.
    Designed specifically for insecticides, not general pesticides or other agrochemicals.

    Molecule must fulfill conditions:

        - 150 <= molecular weight <= 500
        - 0 <= logP <= 5
        - HBD <= 2
        - 1 <= HBA <= 8
        - number of rotatable bonds <= 11

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
    .. [1] `Tice, C.M.,
        "Selecting the right compounds for screening:
        does Lipinski's Rule of 5 for pharmaceuticals apply to agrochemicals?"
        Pest. Manag. Sci., 57: 3-16.
        <https://doi.org/10.1002/1526-4998(200101)57:1\<3::AID-PS269\>3.0.CO;2-6>`_

    Examples
    --------
    >>> from skfp.filters import TiceInsecticidesFilter
    >>> smiles = ["O=C(CC1COc2ccccc2O1)NCCc1ccccc1", "O=C(Nc1cccc(Cl)c1)N1CCCC1", "CNc1nc(N)c([N+](=O)[O-])c(NCCO)n1"]
    >>> filt = TiceInsecticidesFilter()
    >>> filt
    TiceInsecticidesFilter()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['O=C(CC1COc2ccccc2O1)NCCc1ccccc1', 'O=C(Nc1cccc(Cl)c1)N1CCCC1']
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
            150 <= MolWt(mol) <= 500,
            0 <= MolLogP(mol) <= 5,
            CalcNumHBD(mol) <= 2,
            1 <= CalcNumHBA(mol) <= 8,
            CalcNumRotatableBonds(mol) <= 11,
        ]

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
