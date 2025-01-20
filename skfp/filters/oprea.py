from typing import Union

from rdkit.Chem import Mol, rdMolDescriptors

from skfp.bases.base_filter import BaseFilter


class OpreaFilter(BaseFilter):
    """
    Oprea filter.

    Computes Oprea's filter for drug likeness, designed by comparing drug and non-drug
    compounds across multiple datasets [1]_.

    Molecule must fulfill conditions:

    - HBD in range ``[0, 2]``
    - HBA in range ``[2, 9]``
    - number of rotatable bonds in range ``[2, 8]``
    - number of rings in range ``[1, 4]``

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

    verbose : int, default=0
        Controls the verbosity when filtering molecules.

    References
    ----------
    .. [1] `Oprea T. I.
        "Property distribution of drug-related chemical databases"
        J Comput Aided Mol Des. 2000 Mar;14(3):251-64
        <https://pubmed.ncbi.nlm.nih.gov/10756480/>`_

    Examples
    --------
    >>> from skfp.filters import OpreaFilter
    >>> smiles = ["C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O", "CC(=O)Nc1ccc(O)cc1"]
    >>> filt = OpreaFilter()
    >>> filt
    OpreaFilter()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O']
    """

    def __init__(
        self,
        allow_one_violation: bool = False,
        return_indicators: bool = False,
        n_jobs: Union[int, None] = None,
        batch_size: Union[int, None] = None,
        verbose: int = 0,
    ):
        super().__init__(
            allow_one_violation, return_indicators, n_jobs, batch_size, verbose
        )

    def _apply_mol_filter(self, mol: Mol) -> bool:
        rules = [
            0 <= rdMolDescriptors.CalcNumHBD(mol) <= 2,
            2 <= rdMolDescriptors.CalcNumHBA(mol) <= 9,
            2 <= rdMolDescriptors.CalcNumRotatableBonds(mol) <= 8,
            1 <= rdMolDescriptors.CalcNumRings(mol) <= 4,
        ]

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
