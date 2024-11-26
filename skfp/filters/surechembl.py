from typing import Optional, Union

from rdkit.Chem import FilterCatalog, Mol
from rdkit.Chem.rdfiltercatalog import FilterCatalogParams

from skfp.bases.base_filter import BaseFilter


class SureChEMBLFilter(BaseFilter):
    """
    SureChEMBL filter.

    Based on structural alerts, i.e. toxicophores. Filters out compounds likely to be
    toxic. Used at SureChEMBL, and based on the ToxAlerts publication [1]_, so many
    alerts are also found in OChem.

    The original page with rule definitions is available only by the Wayback Machine [2]_.
    They are also available in the RDKit code [3]_.

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
    ----------
    .. [1] `Iurii Sushko et al.
        "ToxAlerts: A Web Server of Structural Alerts for Toxic Chemicals and Compounds
        with Potential Adverse Reactions"
        J. Chem. Inf. Model. 2012, 52, 8, 2310–2316
        <https://pubs.acs.org/doi/10.1021/ci300245q>`_

    .. [2] `Wayback Machine - ChEMBL "Non MedChem-Friendly SMARTS"
        <https://web.archive.org/web/20230111133819/https://www.surechembl.org/knowledgebase/169485-non-medchem-friendly-smarts>`_

    .. [3] `RDKit SureChEMBL filter definitions
        <https://github.com/rdkit/rdkit/blob/e4f4644a89d6446ddebda0bf396fa4335324c41c/Code/GraphMol/FilterCatalog/chembl_surechembl.in>`_

    Examples
    --------
    >>> from skfp.filters import SureChEMBLFilter
    >>> smiles = ["C", "O", "C#CC#C"]
    >>> filt = SureChEMBLFilter()
    >>> filt
    SureChEMBLFilter()

    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['C', 'O']
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
        filter_rules = FilterCatalogParams.FilterCatalogs.CHEMBL_SureChEMBL
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(filter_rules)
        filters = FilterCatalog.FilterCatalog(params)
        return filters

    def _apply_mol_filter(self, mol: Mol) -> bool:
        errors = len(self._filters.GetMatches(mol))
        return not errors or (self.allow_one_violation and errors == 1)
