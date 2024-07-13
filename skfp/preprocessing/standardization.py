from collections.abc import Sequence
from contextlib import nullcontext
from numbers import Integral
from typing import Optional, Union

import joblib
from rdkit.Chem import Mol, SanitizeMol
from rdkit.Chem.MolStandardize.rdMolStandardize import (
    CleanupInPlace,
    FragmentParentInPlace,
)

from skfp.bases import BasePreprocessor
from skfp.utils import ensure_mols, no_rdkit_logs


class MolStandardizer(BasePreprocessor):
    """
    Performs common molecule standardization operations.

    Applies the following cleanup transformations to the inputs:
    - create RDKit Mol objects, if SMILES strings are passed
    - sanitize [1]_ (performs basic validity checks)
    - if `largest_fragment_only`, select the largest fragment for further processing
    - remove hydrogens
    - disconnect metal atoms
    - normalize (transform functional groups to normal form)
    - reionize

    See [1]_ [2]_ [3]_ [4]_ [5]_ for details, rationale and alternatives. Note that
    there is no one-size-fits-all solution, and here we use pretty minimalistic, most
    common steps. This class by design does not allow much parametrization, and for
    custom purposes you should build the pipeline yourself.

    New molecules are always returned, and any set properties are not kept, as this
    should normally be the first step in a pipeline.

    Parameters
    ----------
    largest_fragment_only : bool, default=False
        Whether to select only the largest fragment from each molecule.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized over
        the input molecules. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See Scikit-learn documentation on
        ``n_jobs`` for more details.

    verbose : int, default=0
        Controls the verbosity when standardizing molecules. By default, all warnings are
        turned off.

    References
    ----------
    .. [1] `Gregory Landrum
        "The RDKit Book: Molecular Sanitization"
        <https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization>`_

    .. [2] `RDKit mailing list thread on molecular sanitization
        <https://sourceforge.net/p/rdkit/mailman/rdkit-discuss/thread/CANjYGkS%3Dqz3NbmdhGAEvfOec0o%3DCL%3D43aex%3DT1RGmUDgSb63og%40mail.gmail.com/>`_

    .. [3] `Matt Swain
        "MolVS Standardization"
        <https://molvs.readthedocs.io/en/latest/guide/standardize.html>`_

    .. [4] `Gregory Landrum
        "RSC OpenScience workshop: Chemical structure validation and standardization with the RDKit"
        <https://github.com/greglandrum/RSC_OpenScience_Standardization_202104>`_

    .. [5] `Jean-Paul Ebejer
        "Standardizing a molecule using RDKit"
        <https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolStandardizer
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> standardizer = MolStandardizer()
    >>> standardizer
    MolStandardizer()

    >>> standardizer.transform(smiles)  # doctest: +SKIP
        [<rdkit.Chem.rdchem.Mol object at ...>,
         <rdkit.Chem.rdchem.Mol object at ...>,
         <rdkit.Chem.rdchem.Mol object at ...>,
         <rdkit.Chem.rdchem.Mol object at ...>]
    """

    _parameter_constraints: dict = {
        "largest_fragment_only": ["boolean"],
        "n_jobs": [Integral, None],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        largest_fragment_only: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__()
        self.largest_fragment_only = largest_fragment_only
        # note that parallelization, where possible, is handled at RDKit functions level
        self.n_jobs = n_jobs
        self.verbose = verbose

    def transform(self, X: Sequence[Union[str, Mol]], copy: bool = True) -> list[Mol]:
        self._validate_params()

        n_jobs = joblib.effective_n_jobs(self.n_jobs)

        # here, we perform basic validity check and create new molecules
        # this is too fast to benefit from parallelization
        mols = ensure_mols(X)
        for mol in mols:
            SanitizeMol(mol)

        with nullcontext() if self.verbose else no_rdkit_logs():
            # select the largest ("parent") fragment if needed
            if self.largest_fragment_only:
                FragmentParentInPlace(mols, numThreads=n_jobs)

            # remove Hs, disconnect metals, normalize functional groups, reionize
            CleanupInPlace(mols, numThreads=n_jobs)

        return mols


class HydrogenRemover(BasePreprocessor):
    pass
