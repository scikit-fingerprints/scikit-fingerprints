from typing import Optional, Union

from rdkit.Chem import FilterCatalog, Mol
from rdkit.Chem.rdfiltercatalog import FilterCatalogParams

from skfp.bases.base_filter import BaseFilter


class NIHFilter(BaseFilter):
    """
    NIH filter.

    Designed to filter out molecules containing with undesirable functional groups,
    including reactive functionalities and medicinal chemistry exclusions.

    Rule definitions are available in the supplementary material of the original
    publications [1]_ [2]_ and in RDKit code [3]_.

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
    .. [1] `Ajit Jadhav et al.
        "Quantitative Analyses of Aggregation, Autofluorescence, and Reactivity
        Artifacts in a Screen for Inhibitors of a Thiol Protease"
        J. Med. Chem. 2010, 53, 1, 37-51
        <https://pubs.acs.org/doi/10.1021/jm901070c>`_

    .. [2] `Richard G. Doveston et al.
        "A unified lead-oriented synthesis of over fifty molecular scaffolds"
        Org. Biomol. Chem., 2015,13, 859-865
        <https://pubs.rsc.org/en/content/articlelanding/2015/ob/c4ob02287d>`_

    .. [3] `RDKit NIH filter definitions
        <https://github.com/rdkit/rdkit/blob/e4f4644a89d6446ddebda0bf396fa4335324c41c/Code/GraphMol/FilterCatalog/nih.in>`_

    Examples
    --------
    >>> from skfp.filters import NIHFilter
    >>> smiles = ["C", "C=P"]
    >>> filt = NIHFilter()
    >>> filt
    NIHFilter()

    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['C']
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

    def _load_filters(self) -> FilterCatalog:
        filter_rules = FilterCatalogParams.FilterCatalogs.NIH
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(filter_rules)
        filters = FilterCatalog.FilterCatalog(params)
        return filters

    def _apply_mol_filter(self, mol: Mol) -> bool:
        errors = len(self._filters.GetMatches(mol))
        return not errors or (self.allow_one_violation and errors == 1)
