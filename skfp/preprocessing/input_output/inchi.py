from collections.abc import Sequence
from contextlib import nullcontext
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol, MolFromInchi, MolToInchi

from skfp.bases import BasePreprocessor
from skfp.utils import (
    get_data_from_indices,
    no_rdkit_logs,
    require_mols,
    require_strings,
)


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

    valid_only: bool, default=False
        Whether to return only molecules that were successfully loaded. By default,
        returns ``None`` for molecules that got errors.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    suppress_warnings: bool, default=False
        Whether to suppress warnings and errors on loading molecules.

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
        "valid_only": ["boolean"],
    }

    def __init__(
        self,
        sanitize: bool = True,
        remove_hydrogens: bool = True,
        valid_only: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        suppress_warnings: bool = False,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            batch_size=batch_size,
            suppress_warnings=suppress_warnings,
            verbose=verbose,
        )
        self.sanitize = sanitize
        self.remove_hydrogens = remove_hydrogens
        self.valid_only = valid_only

    def transform(self, X, copy: bool = False) -> list[Mol]:
        """
        Create RDKit ``Mol`` objects from InChI strings. If ``valid_only`` is set to
        True, returns only a subset of molecules which could be successfullyloaded.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing InChI strings.

        copy : bool, default=False
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        X : list of shape (n_samples_conf_gen,)
            List with RDKit ``Mol`` objects.
        """
        X = super().transform(X, copy)
        if self.valid_only:
            X = [mol for mol in X if mol is not None]
        return X

    def transform_x_y(self, X, y, copy: bool = False) -> tuple[list[Mol], np.ndarray]:
        """
        Create RDKit ``Mol`` objects from InChI strings. If ``valid_only`` is set to
        True, returns only a subset of molecules and labels which could be successfully
        loaded.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing InChI strings

        y : np.ndarray of shape (n_samples,)
            Array with labels for molecules.

        copy : bool, default=False
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        X : list of shape (n_samples,)
            List with RDKit ``Mol`` objects.

        y : np.ndarray of shape (n_samples,)
            Array with labels for molecules.
        """
        X = super().transform(X, copy)
        if self.valid_only:
            idxs_to_keep = [idx for idx, mol in X if mol is not None]
            X = get_data_from_indices(X, idxs_to_keep)
            y = y[idxs_to_keep]

        return X, y

    def _transform_batch(self, X: Sequence[str]) -> list[Mol]:
        with no_rdkit_logs() if self.suppress_warnings else nullcontext():
            require_strings(X)
            return [
                MolFromInchi(
                    inchi, sanitize=self.sanitize, removeHs=self.remove_hydrogens
                )
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
        See scikit-learn documentation on ``n_jobs`` for more details.

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
        require_mols(X)
        return [MolToInchi(mol) for mol in X]
