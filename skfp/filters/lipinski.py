from typing import Optional, Union

from rdkit.Chem import Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumLipinskiHBA, CalcNumLipinskiHBD

from skfp.bases.base_filter import BaseFilter


class LipinskiFilter(BaseFilter):
    """
    Lipinski's Rule of 5 (RO5).

    Also known as Pfizer's Rule of 5. It evaluates the drug-likeness of a molecule
    as an orally active drug. Assumes that it should be small and lipophilic.
    Description of the rules can be found in the original publication [1]_.

    Molecule can violate at most one of the rules (conditions):
    - molecular weight <= 500 daltons
    - HBA <= 10
    - HBD <= 5
    - logP <= 5

    Hydrogen bond acceptors (HBA) and donors (HBD) use a simplified definition,
    taking into consideration only oxygen and nitrogen bonds with hydrogen (OH, NH).

    Parameters
    ----------
    allow_one_violation : bool, default=True
        Whether to allow violating one of the rules for a molecule. This makes the
        filter less restrictive, and is the part of the original definition of this
        filter.

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
    ----------
    .. [1] `Christopher A. Lipinski, Franco Lombardo, Beryl W. Dominy, Paul J. Feeney
        "Experimental and computational approaches to estimate solubility and permeability
        in drug discovery and development settings"
        Advanced Drug Delivery Reviews, Volume 23, Issues 1–3, 15 January 1997, Pages 3-25
        <https://www.sciencedirect.com/science/article/pii/S0169409X96004231>`_

    Examples
    --------
    >>> from skfp.filters import LipinskiFilter
    >>> smiles = ["[C-]#N", "CC=O", "O=C(O)c1ccccc1c2ccc(cc2)Cn3c4cc(cc(c4nc3CCC)C)c5nc6ccccc6n5C"]
    >>> filt = LipinskiFilter()
    >>> filt
    LipinskiFilter()

    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['[C-]#N', 'CC=O']
    """

    def __init__(
        self,
        allow_one_violation: bool = True,
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
            MolWt(mol) <= 500,  # molecular weight
            CalcNumLipinskiHBA(mol) <= 10,  # HBA
            CalcNumLipinskiHBD(mol) <= 5,  # HBD
            MolLogP(mol) <= 5,  # logP
        ]
        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
