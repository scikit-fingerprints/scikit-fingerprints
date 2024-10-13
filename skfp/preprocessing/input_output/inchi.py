from collections.abc import Sequence
from typing import Optional, Union

from rdkit.Chem import Mol, MolFromInchi, MolToInchi

from skfp.bases import BasePreprocessor
from skfp.utils.validators import check_mols, check_strings


class MolFromInchiTransformer(BasePreprocessor):
    """
    Creates RDKit ``Mol`` objects from InChI strings.

    For details see RDKit documentation [1]_.

    Parameters
    ----------
    sanitize : bool, default=True
        Whether to perform sanitization [1]_, i.e. basic validity checks, on created
        molecules.

    remove_hydrogens : bool, default=True
        Remove explicit hydrogens from the molecule where possible, using RDKit
        implicit hydrogens instead.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See Scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when processing molecules.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar..

    References
    ----------
    .. [1] `RDKit InChI documentation
        <https://www.rdkit.org/docs/source/rdkit.Chem.inchi.html>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolFromInchiTransformer
    >>> inchi_list = ["1S/H2O/h1H2", "1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3"]
    >>> mol_from_inchi = MolFromInchiTransformer()
    >>> mol_from_inchi
    MolFromInchiTransformer()

    >>> mol_from_inchi.transform(inchi_list)  # doctest: +SKIP
        [<rdkit.Chem.rdchem.Mol>,
         <rdkit.Chem.rdchem.Mol>]
    """

    _parameter_constraints: dict = {
        **BasePreprocessor._parameter_constraints,
        "sanitize": ["boolean"],
        "remove_hydrogens": ["boolean"],
    }

    def __init__(
        self,
        sanitize: bool = True,
        remove_hydrogens: bool = True,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.sanitize = sanitize
        self.remove_hydrogens = remove_hydrogens

    def _transform_batch(self, X: Sequence[str]) -> list[Mol]:
        check_strings(X)
        return [
            MolFromInchi(inchi, sanitize=self.sanitize, removeHs=self.remove_hydrogens)
            for inchi in X
        ]


class MolToInchiTransformer(BasePreprocessor):
    """
    Creates InChI strings from RDKit ``Mol`` objects.

    For details see RDKit documentation [1]_.

    Parameters
    ----------
    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See Scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when processing molecules.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    References
    ----------
    .. [1] `RDKit InChI documentation
        <https://www.rdkit.org/docs/source/rdkit.Chem.inchi.html>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolFromInchiTransformer, MolToInchiTransformer
    >>> inchi_list = ["InChI=1S/H2O/h1H2", "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3"]
    >>> mol_from_inchi = MolFromInchiTransformer()
    >>> mol_to_inchi = MolToInchiTransformer()
    >>> mol_to_inchi
    MolToInchiTransformer()

    >>> mols = mol_from_inchi.transform(inchi_list)
    >>> mol_to_inchi.transform(mols)
    ['InChI=1S/H2O/h1H2', 'InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3']
    """

    def __init__(
        self,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def _transform_batch(self, X: Sequence[Mol]) -> list[str]:
        check_mols(X)
        return [MolToInchi(mol) for mol in X]
