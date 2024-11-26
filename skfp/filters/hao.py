from typing import Optional, Union

from rdkit.Chem import Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD, CalcNumRotatableBonds

from skfp.bases.base_filter import BaseFilter


class HaoFilter(BaseFilter):
    """
    Hao rule for pesticides.

    Designed to describe physicochemical properties of pesticides [1]_.
    Can be used in general pesticide design.

    Molecule must fulfill conditions:

        - molecular weight <= 435
        - logP <= 6
        - HBD <= 2
        - HBA <= 6
        - number of rotatable bonds <= 9
        - number of aromatic bonds <= 17


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
    .. [1] `Hao, G., Dong, Q. and Yang, G.,
        "A Comparative Study on the Constitutive Properties of Marketed Pesticides."
        Mol. Inf., 30: 614-622.
        <https://doi.org/10.1002/minf.201100020>`_

    Examples
    ----------
    >>> from skfp.filters import HaoFilter
    >>> smiles = ["CN(C)c1ccc(C=Cc2cc[n+](C)c3ccccc23)cc1","c1cnc2c(c1)ccc1cccnc12",\
    "Cn1c(SSc2ccc(-c3cccnc3)n2C)ccc1-c1cccnc1"]
    >>> filt = HaoFilter()
    >>> filt
    HaoFilter()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['CN(C)c1ccc(C=Cc2cc[n+](C)c3ccccc23)cc1', 'c1cnc2c(c1)ccc1cccnc12']
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
        aromatic_bond_count = sum(
            bond.GetBondType() == BondType.AROMATIC for bond in mol.GetBonds()
        )
        rules = [
            MolWt(mol) <= 435,
            MolLogP(mol) <= 6,
            CalcNumHBD(mol) <= 2,
            CalcNumHBA(mol) <= 6,
            CalcNumRotatableBonds(mol) <= 9,
            aromatic_bond_count <= 17,
        ]

        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)
