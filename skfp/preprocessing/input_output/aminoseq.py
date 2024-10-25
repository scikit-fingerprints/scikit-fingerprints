from collections.abc import Sequence
from contextlib import nullcontext
from typing import Optional

from rdkit.Chem import Mol, MolFromFASTA
from sklearn.utils._param_validation import Options

from skfp.bases import BasePreprocessor
from skfp.utils import no_rdkit_logs
from skfp.utils.validators import check_strings


class MolFromAminoseqTransformer(BasePreprocessor):
    """
    Creates RDKit ``Mol`` objects from amino-acid sequence strings.

    Inputs are either sequences in FASTA format [1]_, or plain strings with
    amino-acid sequences.

    Parameters
    ----------
    sanitize : bool, default=True
        Whether to perform sanitization, i.e. basic validity checks, on created
        molecules. For details see RDKit documentation [2]_.

    flavor : int, default=0
        Type of molecule. See RDKit documentation [3]_ for more details.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See Scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    suppress_warnings: bool, default=False
        Whether to suppress warnings and errors on loading molecules.

    verbose : int, default=0
        Controls the verbosity when processing molecules.

    References
    ----------
    .. [1] `Lipman DJ, Pearson WR.
        "Rapid and sensitive protein similarity searches."
        Science. 1985 Mar 22; 227(4693):1435-41.
        <https://pubmed.ncbi.nlm.nih.gov/2983426/>`_

    .. [2] `Gregory Landrum
        "The RDKit Book: Molecular Sanitization"
        <https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization>`_

    .. [3] `RDKit documentation - MolFromFASTA
        <https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.MolFromFASTA>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolFromAminoseqTransformer
    >>> sequences = ["KWLRRVWRWWR","FLPAIGRVLSGIL","ILGKLLSTAWGLLSKL",]
    >>> mol_from_aminoseq = MolFromAminoseqTransformer()
    >>> mol_from_aminoseq
    MolFromAminoseqTransformer()

    >>> mol_from_aminoseq.transform(sequences)  # doctest: +SKIP
        [<rdkit.Chem.rdchem.Mol>,
         <rdkit.Chem.rdchem.Mol>,
         <rdkit.Chem.rdchem.Mol>,
         <rdkit.Chem.rdchem.Mol>]
    """

    _parameter_constraints: dict = {
        **BasePreprocessor._parameter_constraints,
        "sanitize": ["boolean"],
        "flavor": [Options(int, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})],
    }

    def __init__(
        self,
        sanitize: bool = True,
        flavor: int = 0,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        suppress_warnings: bool = False,
        verbose: int = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            batch_size=batch_size,
            suppress_warnings=suppress_warnings,
            verbose=verbose,
        )
        self.sanitize = sanitize
        self.flavor = flavor

    def transform(self, X, copy: bool = False) -> list[Mol]:
        """
        Create RDKit ``Mol`` objects from amino-acid sequence strings. If ``valid_only``
        is set toTrue, returns only a subset of molecules which could be successfully
        loaded.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing amino-acid sequence strings.

        copy : bool, default=False
            Unused, kept for Scikit-learn compatibility.

        Returns
        -------
        X : list of shape (n_samples_conf_gen,)
            List with RDKit ``Mol`` objects.
        """
        return super().transform(X, copy)

    def _transform_batch(self, X: Sequence[str]) -> list[Mol]:
        with no_rdkit_logs() if self.suppress_warnings else nullcontext():
            check_strings(X)
            return [
                MolFromFASTA(fst, sanitize=self.sanitize, flavor=self.flavor)
                for fst in X
            ]
