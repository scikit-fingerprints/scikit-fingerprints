from numbers import Integral
from typing import Optional, Union

from rdkit.Chem import Mol
from rdkit.Chem.Descriptors import MolWt
from sklearn.utils._param_validation import Interval, InvalidParameterError

from skfp.bases.base_filter import BaseFilter


class MolecularWeightFilter(BaseFilter):
    """
    Molecular weight filter.

    Filters out molecules with mass in Daltons outside the given range. Provided
    ``min_weight`` and ``max_weight`` are inclusive on both sides.

    Parameters
    ----------
    min_weight : int, default=0
        Minimal weight in Daltons.

    max_weight : int, default=1000
        Maximal weight in Daltons.

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

    Examples
    --------
    >>> from skfp.filters import MolecularWeightFilter
    >>> smiles = ["C", "O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
    >>> filt = MolecularWeightFilter(max_weight=100)
    >>> filt
    MolecularWeightFilter(max_weight=100)

    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['C', 'O']
    """

    _parameter_constraints: dict = {
        **BaseFilter._parameter_constraints,
        "min_weight": [Interval(Integral, 0, None, closed="left")],
        "max_weight": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        min_weight: int = 0,
        max_weight: int = 1000,
        return_indicators: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            return_indicators=return_indicators,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.min_weight = min_weight
        self.max_weight = max_weight

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.max_weight < self.min_weight:
            raise InvalidParameterError(
                f"The max_weight parameter of {self.__class__.__name__} must be "
                f"greater or equal to min_weight, got: "
                f"min_weight={self.min_weight}, max_weight={self.max_weight}"
            )

    def _apply_mol_filter(self, mol: Mol) -> bool:
        return self.min_weight <= MolWt(mol) <= self.max_weight
