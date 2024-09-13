from typing import Optional

from rdkit.Chem import Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import (
    CalcNumHBA,
    CalcNumHBD,
    CalcNumRotatableBonds,
    CalcTPSA,
)

from skfp.bases.base_filter import BaseFilter


class RuleOf3(BaseFilter):
    """
    Rule optimised to search for fragment-based lead-like compounds with desired properties.
    It was described in [1]

    Acording to this rule fragment-based lead-like compound should meet following criteria:
    - molecular weight <= 300 daltons
    - HBA <= 3
    - HBD <= 3
    - logP <= 3
    - TPSA <= 60 (only in extemded version of this rule)
    - number of rotatable bonds <= 3 (only in extemded version of this rule)
    Parameters
    ----------
    allow_one_violation : bool, default=True
        Whether to allow violating one of the rules for a molecule. This makes the
        filter less restrictive, and is the part of the original definition of this
        filter.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform_x_y` and
        :meth:`transform` are parallelized over the input molecules. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See Scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int, default=0
        Controls the verbosity when generating conformers.

    extended : bool,default=False
         results in the application of an extended version of this rule
         i.e. apply TPSA and rotatable bonds filtering
    References
    -----------
    .. [1] 'Congreve, M., Carr, R., Murray, C., & Jhoti, H. (2003).
        A ‘Rule of Three’ for fragment-based lead discovery? Drug Discovery Today, 8(19), 876–877.
        doi:10.1016/S1359-6446(03)02831-9'_

     Examples
    ----------
    >>> from skfp.preprocessing import RuleOf2
    >>> smiles = ["C=CCNC(=S)NCc1ccccc1OC", "C=CCOc1ccc(Br)cc1/C=N/O", "C=CCSc1ncccc1C(=O)O"]
    >>> filt = RuleOf3()
    >>> filt
    RuleOf3()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ["C=CCNC(=S)NCc1ccccc1OC", "C=CCOc1ccc(Br)cc1/C=N/O", "C=CCSc1ncccc1C(=O)O"]
    >>> filt = RuleOf3(extended=True)
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ["C=CCNC(=S)NCc1ccccc1OC", "C=CCOc1ccc(Br)cc1/C=N/O"]
    """

    def __init__(
        self,
        allow_one_violation: bool = False,
        return_indicators: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
        extended: bool = False,
    ):

        super().__init__(
            allow_one_violation=allow_one_violation,
            return_indicators=return_indicators,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.extended = extended

    def _apply_mol_filter(self, mol: Mol) -> bool:

        rules = [
            MolWt(mol) <= 300,
            CalcNumHBA(mol) <= 3,
            CalcNumHBD(mol) <= 3,
            MolLogP(mol) <= 3,
        ]
        if self.extended:
            rules += [CalcNumRotatableBonds(mol) <= 3, CalcTPSA(mol) <= 60]
        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)


"""ro2 = RuleOf3(return_indicators=True, extended=True)
smiles = ['C=CCOc1ccc(Cl)cc1C(=O)O', 'C=CCSc1ncccc1C(=O)O', 'c1ccc(CCCNc2ncccn2)cc1' ]
print(ro2.transform(smiles))"""
