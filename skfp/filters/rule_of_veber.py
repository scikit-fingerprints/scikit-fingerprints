from typing import Union

from rdkit.Chem import Mol, rdMolDescriptors

from skfp.bases.base_filter import BaseFilter


class RuleOfVeberFilter(BaseFilter):
    """
    Rule of Veber.

    The rule is designed to identify molecules likely to exhibit good oral
    bioavailability, based on the number of rotatable bonds and the topological
    polar surface area (TPSA) [1]_.

    Molecule must fulfill conditions:

    - number of rotatable bonds <= 10
    - TPSA <= 140

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
    .. [1] `Veber, D.F, Johnson S. R., Cheng H., Smith B. R., Ward K. W. , Kopple K. D.
        "Molecular Properties That Influence the Oral Bioavailability of Drug Candidates."
        J Med Chem. 2002 Jun 6;45(12):2615-23.
        <https://pubmed.ncbi.nlm.nih.gov/12036371/>`_

    Examples
    --------
    >>> from skfp.filters import RuleOfVeberFilter
    >>> smiles = ["CC=O", "CC(C)[C@@H](CC1=CC(=C(C=C1)OC)OCCCOC)C[C@@H]([C@H](C[C@@H](C(C)C)C(=O)NCC(C)(C)C(=O)N)O)N"]
    >>> filt = RuleOfVeberFilter()
    >>> filt
    RuleOfVeberFilter()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['CC=O']
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
            rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10,
            rdMolDescriptors.CalcTPSA(mol) <= 140,
        ]

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
