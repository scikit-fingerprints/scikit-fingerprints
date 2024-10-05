from typing import Union

from rdkit.Chem import Mol, rdMolDescriptors

from skfp.bases.base_filter import BaseFilter


class RuleOfVeber(BaseFilter):
    """
    Rule of Veber

    Rule designed to look for molecules that are likely to exhibit
    good oral bioavailability. Described in [1]_.

    Molecule must fulfill conditions:

    - number of rotatable bonds >= 10
    - polar surface area (PSA) >= 140

    Parameters
    ----------
    allow_one_violation : bool, default=False
        Whether to allow violating one of the rules for a molecule. This makes the
        filter less restrictive.

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

    References
    -----------
    .. [1] `Veber, D.F, Johnson S. R., Cheng H., Smith B. R., Ward K. W. , Kopple K. D.
        "Molecular properties that influence the oral bioavailability of drug candidates."
        J Med Chem. 2002 Jun 6;45(12):2615-23.
        <https://pubmed.ncbi.nlm.nih.gov/12036371/>`_

    Examples
    ----------
    >>> from skfp.preprocessing import RuleOfVeber
    >>> smiles = ["[C-]#N", "CC=O", "CC1=CC(=C(C=C1)C(=O)N(C(=O)N1CCCCC1)C)C(=O)O"]
    >>> filt = RuleOfVeber()
    >>> filt
    RuleOfVeber()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ["[C-]#N", "CC=O"]
    """

    def __init__(
        self,
        allow_one_violation: bool = True,
        return_indicators: bool = False,
        n_jobs: Union[int, None] = None,
        batch_size: Union[int, None] = None,
        verbose: int = 0,
    ):
        super().__init__(
            allow_one_violation, return_indicators, n_jobs, batch_size, verbose
        )
        self.max_rotatable_bonds = 10
        self.max_psa = 140.0

    def _apply_mol_filter(self, mol: Mol) -> bool:
        num_rotatable_bonds: int = rdMolDescriptors.CalcNumRotatableBonds(mol)
        psa: float = rdMolDescriptors.CalcTPSA(mol)

        passes_rotatable_bonds = num_rotatable_bonds <= self.max_rotatable_bonds
        passes_psa = psa <= self.max_psa

        if self.allow_one_violation:
            return passes_rotatable_bonds or passes_psa

        return passes_rotatable_bonds and passes_psa
