from typing import Optional, Union

from rdkit.Chem import FilterCatalog, Mol
from rdkit.Chem.rdfiltercatalog import FilterCatalogParams

from skfp.bases.base_filter import BaseFilter


class BrenkFilter(BaseFilter):
    """
    Brenk filter.

    Designed to filter out molecules containing substructures with undesirable
    pharmacokinetics or toxicity, e.g. sulfates, phosphates, nitro groups. Resulting
    set should be reasonable lead-like molecules for optimization campaigns and HTS.

    Rule definitions are available in the supplementary material of the original
    publication [1]_ and in RDKit code [2]_.

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
    .. [1] `Ruth Brenk et al.
        "Lessons Learnt from Assembling Screening Libraries for Drug Discovery for Neglected Diseases"
        ChemMedChem 3:435-444 (2008)
        <https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/cmdc.200700139>`_

    .. [2] `RDKit Brenk filter definitions
        <https://github.com/rdkit/rdkit/blob/e4f4644a89d6446ddebda0bf396fa4335324c41c/Code/GraphMol/FilterCatalog/brenk.in>`_

    Examples
    --------
    >>> from skfp.filters import BrenkFilter
    >>> smiles = ["C", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1cc([NH2])ccc1"]
    >>> filt = BrenkFilter()
    >>> filt
    BrenkFilter()

    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['C', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
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
        self._filters = self._load_filters()

    @staticmethod
    def _load_filters() -> FilterCatalog:
        filter_rules = FilterCatalogParams.FilterCatalogs.BRENK
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(filter_rules)
        filters = FilterCatalog.FilterCatalog(params)
        return filters

    def _apply_mol_filter(self, mol: Mol) -> bool:
        errors = len(self._filters.GetMatches(mol))
        return not errors or (self.allow_one_violation and errors == 1)
