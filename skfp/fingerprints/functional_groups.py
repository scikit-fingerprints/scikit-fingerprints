from collections.abc import Sequence
from inspect import getmembers, isfunction
from typing import Optional, Union

import numpy as np
import rdkit.Chem.Fragments
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class FunctionalGroupsFingerprint(BaseFingerprintTransformer):
    """
    Functional groups fingerprint.

    The implementation uses RDKit. This is a substructure, descriptor fingerprint,
    checking occurrences of 85 functional groups (also known as fragments) available
    in RDKit [1]_.

    Parameters
    ----------
    count : bool, default=False
        Whether to return binary (bit) features, or their counts.

    sparse : bool, default=False
        Whether to return dense NumPy array, or sparse SciPy CSR array.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when computing fingerprints.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    Attributes
    ----------
    n_features_out : int = 85
        Number of output features, size of fingerprints.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    References
    ----------
    .. [1] RDKit, rdkit.Chem.Fragments module
        https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html

    Examples
    --------
    >>> from skfp.fingerprints import FunctionalGroupsFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O", "CCO", "CCN"]
    >>> fp = FunctionalGroupsFingerprint()
    >>> fp
    FunctionalGroupsFingerprint()
    >>> X = fp.transform(smiles)
    >>> X  # doctest: +SKIP
    array([[0, 0, 0, ..., 1, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 1, 0, 0],
           [0, 0, 0, ..., 1, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    """

    def __init__(
        self,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            n_features_out=85,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        """
        Get fingerprint output feature names. They are descriptions of RDKit
        functional groups (fragments) - see `<https://rdkit.org/docs/source/rdkit.Chem.Fragments.html>`_
        for details.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Names of the RDKit function groups.
        """
        return np.asarray(
            [
                "aliphatic carboxylic acids",
                "aliphatic hydroxyl groups",
                "aliphatic hydroxyl groups excluding tert-OH",
                "N functional groups attached to aromatics",
                "Aromatic carboxylic acide",
                "aromatic nitrogens",
                "aromatic amines",
                "aromatic hydroxyl groups",
                "carboxylic acids",
                "carboxylic acids",
                "carbonyl O",
                "carbonyl O, excluding COOH",
                "thiocarbonyl",
                "C(OH)CCN-Ctert-alkyl or  C(OH)CCNcyclic",
                "Imines",
                "Tertiary amines",
                "Secondary amines",
                "Primary amines",
                "hydroxylamine groups",
                "XCCNR groups",
                "tert-alicyclic amines (no heteroatoms, not quinine-like bridged N)",
                "H-pyrrole nitrogens",
                "thiol groups",
                "aldehydes",
                "alkyl carbamates (subject to hydrolysis)",
                "alkyl halides",
                "allylic oxidation sites excluding steroid dienone",
                "amides",
                "amidine groups",
                "anilines",
                "aryl methyl sites for hydroxylation",
                "azide groups",
                "azo groups",
                "barbiturate groups",
                "benzene rings",
                "benzodiazepines with no additional fused rings",
                "Bicyclic",
                "diazo groups",
                "dihydropyridines",
                "epoxide rings",
                "esters",
                "ether oxygens (including phenoxy)",
                "furan rings",
                "guanidine groups",
                "halogens",
                "hydrazine groups",
                "hydrazone groups",
                "imidazole rings",
                "imide groups",
                "isocyanates",
                "isothiocyanates",
                "ketones",
                "ketones excluding diaryl, a,b-unsat. dienones, heteroatom on Calpha",
                "beta lactams",
                "cyclic esters (lactones)",
                "methoxy groups -OCH3",
                "morpholine rings",
                "nitriles",
                "nitro groups",
                "nitro benzene ring substituents",
                "non-ortho nitro benzene ring substituents",
                "nitroso groups, excluding NO2",
                "oxazole rings",
                "oxime groups",
                "para-hydroxylation sites",
                "phenols",
                "phenolic OH excluding ortho intramolecular Hbond substituents",
                "phosphoric acid groups",
                "phosphoric ester groups",
                "piperdine rings",
                "piperzine rings",
                "primary amides",
                "primary sulfonamides",
                "pyridine rings",
                "quaternary nitrogens",
                "thioether",
                "sulfonamides",
                "sulfone groups",
                "terminal acetylenes",
                "tetrazole rings",
                "thiazole rings",
                "thiocyanates",
                "thiophene rings",
                "unbranched alkanes of at least 4 members (excludes halogenated alkanes)",
                "urea groups",
            ],
            dtype=object,
        )

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute functional groups fingerprints.

        Parameters
        ----------
        X : {sequence of str or Mol}
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Whether to copy input data.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 85)
            Transformed data.
        """
        return super().transform(X, copy=copy)

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        X = ensure_mols(X)

        func_groups_functions = [
            function
            for name, function in getmembers(rdkit.Chem.Fragments, isfunction)
            if name.startswith("fr_")
        ]
        X = np.array([[fun(mol) for fun in func_groups_functions] for mol in X])

        if not self.count:
            X = X > 0

        dtype = np.uint32 if self.count else np.uint8
        return csr_array(X, dtype=dtype) if self.sparse else X.astype(dtype)
