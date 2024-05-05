from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import rdkit.Chem.Fragments
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import ensure_mols


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
        See Scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int, default=0
        Controls the verbosity when computing fingerprints.

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
    >>> X
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
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=85,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        X = ensure_mols(X)

        X = np.array([self._get_functional_groups_counts(mol) for mol in X])

        if not self.count:
            X = X > 0

        dtype = np.uint32 if self.count else np.uint8
        return csr_array(X, dtype=dtype) if self.sparse else X.astype(dtype)

    def _get_functional_groups_counts(self, mol: Mol) -> list[int]:
        func_groups_counts = [
            func_group(mol)
            for func_group in [
                rdkit.Chem.Fragments.fr_Al_COO,
                rdkit.Chem.Fragments.fr_Al_OH,
                rdkit.Chem.Fragments.fr_Al_OH_noTert,
                rdkit.Chem.Fragments.fr_ArN,
                rdkit.Chem.Fragments.fr_Ar_COO,
                rdkit.Chem.Fragments.fr_Ar_N,
                rdkit.Chem.Fragments.fr_Ar_NH,
                rdkit.Chem.Fragments.fr_Ar_OH,
                rdkit.Chem.Fragments.fr_COO,
                rdkit.Chem.Fragments.fr_COO2,
                rdkit.Chem.Fragments.fr_C_O,
                rdkit.Chem.Fragments.fr_C_O_noCOO,
                rdkit.Chem.Fragments.fr_C_S,
                rdkit.Chem.Fragments.fr_HOCCN,
                rdkit.Chem.Fragments.fr_Imine,
                rdkit.Chem.Fragments.fr_NH0,
                rdkit.Chem.Fragments.fr_NH1,
                rdkit.Chem.Fragments.fr_NH2,
                rdkit.Chem.Fragments.fr_N_O,
                rdkit.Chem.Fragments.fr_Ndealkylation1,
                rdkit.Chem.Fragments.fr_Ndealkylation2,
                rdkit.Chem.Fragments.fr_Nhpyrrole,
                rdkit.Chem.Fragments.fr_SH,
                rdkit.Chem.Fragments.fr_aldehyde,
                rdkit.Chem.Fragments.fr_alkyl_carbamate,
                rdkit.Chem.Fragments.fr_alkyl_halide,
                rdkit.Chem.Fragments.fr_allylic_oxid,
                rdkit.Chem.Fragments.fr_amide,
                rdkit.Chem.Fragments.fr_amidine,
                rdkit.Chem.Fragments.fr_aniline,
                rdkit.Chem.Fragments.fr_aryl_methyl,
                rdkit.Chem.Fragments.fr_azide,
                rdkit.Chem.Fragments.fr_azo,
                rdkit.Chem.Fragments.fr_barbitur,
                rdkit.Chem.Fragments.fr_benzene,
                rdkit.Chem.Fragments.fr_benzodiazepine,
                rdkit.Chem.Fragments.fr_bicyclic,
                rdkit.Chem.Fragments.fr_diazo,
                rdkit.Chem.Fragments.fr_dihydropyridine,
                rdkit.Chem.Fragments.fr_epoxide,
                rdkit.Chem.Fragments.fr_ester,
                rdkit.Chem.Fragments.fr_ether,
                rdkit.Chem.Fragments.fr_furan,
                rdkit.Chem.Fragments.fr_guanido,
                rdkit.Chem.Fragments.fr_halogen,
                rdkit.Chem.Fragments.fr_hdrzine,
                rdkit.Chem.Fragments.fr_hdrzone,
                rdkit.Chem.Fragments.fr_imidazole,
                rdkit.Chem.Fragments.fr_imide,
                rdkit.Chem.Fragments.fr_isocyan,
                rdkit.Chem.Fragments.fr_isothiocyan,
                rdkit.Chem.Fragments.fr_ketone,
                rdkit.Chem.Fragments.fr_ketone_Topliss,
                rdkit.Chem.Fragments.fr_lactam,
                rdkit.Chem.Fragments.fr_lactone,
                rdkit.Chem.Fragments.fr_methoxy,
                rdkit.Chem.Fragments.fr_morpholine,
                rdkit.Chem.Fragments.fr_nitrile,
                rdkit.Chem.Fragments.fr_nitro,
                rdkit.Chem.Fragments.fr_nitro_arom,
                rdkit.Chem.Fragments.fr_nitro_arom_nonortho,
                rdkit.Chem.Fragments.fr_nitroso,
                rdkit.Chem.Fragments.fr_oxazole,
                rdkit.Chem.Fragments.fr_oxime,
                rdkit.Chem.Fragments.fr_para_hydroxylation,
                rdkit.Chem.Fragments.fr_phenol,
                rdkit.Chem.Fragments.fr_phenol_noOrthoHbond,
                rdkit.Chem.Fragments.fr_phos_acid,
                rdkit.Chem.Fragments.fr_phos_ester,
                rdkit.Chem.Fragments.fr_piperdine,
                rdkit.Chem.Fragments.fr_piperzine,
                rdkit.Chem.Fragments.fr_priamide,
                rdkit.Chem.Fragments.fr_prisulfonamd,
                rdkit.Chem.Fragments.fr_pyridine,
                rdkit.Chem.Fragments.fr_quatN,
                rdkit.Chem.Fragments.fr_sulfide,
                rdkit.Chem.Fragments.fr_sulfonamd,
                rdkit.Chem.Fragments.fr_sulfone,
                rdkit.Chem.Fragments.fr_term_acetylene,
                rdkit.Chem.Fragments.fr_tetrazole,
                rdkit.Chem.Fragments.fr_thiazole,
                rdkit.Chem.Fragments.fr_thiocyan,
                rdkit.Chem.Fragments.fr_thiophene,
                rdkit.Chem.Fragments.fr_unbrch_alkane,
                rdkit.Chem.Fragments.fr_urea,
            ]
        ]
        return func_groups_counts
