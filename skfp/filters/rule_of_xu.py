from typing import Union

from rdkit.Chem import Mol, rdMolDescriptors

from skfp.bases.base_filter import BaseFilter


class RuleOfXuFilter(BaseFilter):
    """
    Rule of Xu.

    This rule is designed to identify drug-like molecules [1]_.

    Molecule must fulfill conditions:

    - HBD <= 5
    - HBA <= 10
    - number of rotatable bonds in range ``[2, 35]``
    - number of rings in range ``[1, 7]``
    - number of heavy atoms in range ``[10, 50]``

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
    .. [1] `Xu, J., Stevenson, J.
        "Drug-like Index: A New Approach To Measure Drug-like Compounds and Their Diversity."
        J Chem Inf Comput Sci. 2000 Sep-Oct;40(5):1177-87
        <https://pubmed.ncbi.nlm.nih.gov/11045811/>`_

    Examples
    ----------
    >>> from skfp.filters import RuleOfXuFilter
    >>> smiles = ["CCO", "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O"]
    >>> filt = RuleOfXuFilter()
    >>> filt
    RuleOfXuFilter()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O']
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
            rdMolDescriptors.CalcNumHBD(mol) <= 5,
            rdMolDescriptors.CalcNumHBA(mol) <= 10,
            2 <= rdMolDescriptors.CalcNumRotatableBonds(mol) <= 35,
            1 <= rdMolDescriptors.CalcNumRings(mol) <= 7,
            10 <= rdMolDescriptors.CalcNumHeavyAtoms(mol) <= 50,
        ]

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
