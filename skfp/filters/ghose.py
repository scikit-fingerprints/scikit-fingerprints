from typing import Optional, Union

from rdkit.Chem import Mol
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcCrippenDescriptors, CalcNumAtoms

from skfp.bases.base_filter import BaseFilter


class GhoseFilter(BaseFilter):
    """
    Ghose Filter.

    Used to searching for drug-like molecules [1]_.

    Molecule must fulfill conditions:

        - 160 <= molecular weight <= 400
        - -0.4 <= logP <= 5.6
        - 20 <= number of atoms <= 70
        - 40 <= molar refractivity <= 130

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
    .. [1] `Ghose, A. K., Viswanadhan, V. N., & Wendoloski, J. J.
        "A Knowledge-Based Approach in Designing Combinatorial or Medicinal Chemistry Libraries for Drug Discovery. 1.
        A Qualitative and Quantitative Characterization of Known Drug Databases."
        Journal of Combinatorial Chemistry, 1(1), 55-68.
        <https://doi.org/10.1021/cc9800071>`_

    Examples
    --------
    >>> from skfp.filters import GhoseFilter
    >>> smiles = ["CC(=O)C1=C(O)C(=O)N(CCc2c[nH]c3ccccc23)C1c1ccc(C)cc1", "CC(=O)c1c(C)n(CC2CCCO2)c2ccc(O)cc12",\
    "CC(=O)c1c(C(C)=O)c(C)n(CCCCn2c(C)c(C(C)=O)c(C(C)=O)c2C)c1C"]
    >>> filt = GhoseFilter()
    >>> filt
    GhoseFilter()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['CC(=O)C1=C(O)C(=O)N(CCc2c[nH]c3ccccc23)C1c1ccc(C)cc1', 'CC(=O)c1c(C)n(CC2CCCO2)c2ccc(O)cc12']
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
        logp, mr = CalcCrippenDescriptors(mol)
        rules = [
            160 <= MolWt(mol) <= 400,
            40 <= mr <= 130,
            20 <= CalcNumAtoms(mol) <= 70,
            -0.4 <= logp <= 5.6,
        ]
        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
