from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseSubstructureFingerprint


class GhoseCrippenFingerprint(BaseSubstructureFingerprint):
    """
    Ghose-Crippen fingerprint.

    A substructure fingerprint based on 110 atom types proposed by Ghose and
    Crippen [1]_ [2]_. They are defined for carbon, hydrogen, oxygen, nitrogen, sulfur,
    and halogens, and originally applied for predicting molar refractivities and logP.

    RDKit SMARTS patterns definitions are used [3]_.

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
    n_features_out : int = 110
        Number of output features, size of fingerprints.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    References
    ----------
    .. [1] `Arup K. Ghose and Gordon M. Crippen
        "Atomic Physicochemical Parameters for Three-Dimensional Structure-Directed
        Quantitative Structure-Activity Relationships I. Partition Coefficients as a Measure of Hydrophobicity"
        Journal of Computational Chemistry 7.4 (1986): 565-577.
        <https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.540070419>`_

    .. [2] `Arup K. Ghose and Gordon M. Crippen
        "Atomic physicochemical parameters for three-dimensional-structure-directed
        quantitative structure-activity relationships. 2. Modeling dispersive and hydrophobic interactions"
        J. Chem. Inf. Comput. Sci. 1987, 27, 1, 21-35
        <https://pubs.acs.org/doi/10.1021/ci00053a005>`_

    .. [3] `<https://github.com/rdkit/rdkit/blob/5d034e37331c2604bf3e247b94be35b519e62216/Data/Crippen.txt>`_

    Examples
    --------
    >>> from skfp.fingerprints import GhoseCrippenFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = GhoseCrippenFingerprint()
    >>> fp
    GhoseCrippenFingerprint()

    >>> fp.transform(smiles)  # doctest: +ELLIPSIS
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 1, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 1, 0, ..., 0, 0, 0]],
          dtype=uint8)
    """

    def __init__(
        self,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        # copyright notice for SMARTS patterns for Ghose-Crippen:
        #
        #  Copyright (C) 2002-2012 Greg Landrum and Rational Discovery LLC
        #   This file is part of the RDKit.
        #   The contents are covered by the terms of the BSD license
        #   which is included in the file license.txt, found at the root
        #   of the RDKit source tree.
        #
        # we include copy of that license in skfp/fingerprints/data/RDKit_license.txt
        patterns = [
            "[CH4]",
            "[CH3]C",
            "[CH2](C)C",
            "[CH](C)(C)C",
            "[C](C)(C)(C)C",
            "[CH3][N,O,P,S,F,Cl,Br,I]",
            "[CH2X4]([N,O,P,S,F,Cl,Br,I])[A;!#1]",
            "[CH1X4]([N,O,P,S,F,Cl,Br,I])([A;!#1])[A;!#1]",
            "[CH0X4]([N,O,P,S,F,Cl,Br,I])([A;!#1])([A;!#1])[A;!#1]",
            "[C]=[!C;A;!#1]",
            "[CH2]=C",
            "[CH1](=C)[A;!#1]",
            "[CH0](=C)([A;!#1])[A;!#1]",
            "[C](=C)=C",
            "[CX2]#[A;!#1]",
            "[CH3]c",
            "[CH3]a",
            "[CH2X4]a",
            "[CHX4]a",
            "[CH0X4]a",
            "[cH0]-[A;!C;!N;!O;!S;!F;!Cl;!Br;!I;!#1]",
            "[c][#9]",
            "[c][#17]",
            "[c][#35]",
            "[c][#53]",
            "[cH]",
            "[c](:a)(:a):a",
            "[c](:a)(:a)-a",
            "[c](:a)(:a)-C",
            "[c](:a)(:a)-N",
            "[c](:a)(:a)-O",
            "[c](:a)(:a)-S",
            "[c](:a)(:a)=[C,N,O]",
            "[C](=C)(a)[A;!#1]",
            "[C](=C)(c)a",
            "[CH1](=C)a",
            "[C]=c",
            "[CX4][A;!C;!N;!O;!P;!S;!F;!Cl;!Br;!I;!#1]",
            "[#6]",
            "[#1][#6,#1]",
            "[#1]O[CX4,c]",
            "[#1]O[!C;!N;!O;!S]",
            "[#1][!C;!N;!O]",
            "[#1][#7]",
            "[#1]O[#7]",
            "[#1]OC=[#6,#7,O,S]",
            "[#1]O[O,S]",
            "[#1]",
            "[NH2+0][A;!#1]",
            "[NH+0]([A;!#1])[A;!#1]",
            "[NH2+0]a",
            "[NH1+0]([!#1;A,a])a",
            "[NH+0]=[!#1;A,a]",
            "[N+0](=[!#1;A,a])[!#1;A,a]",
            "[N+0]([A;!#1])([A;!#1])[A;!#1]",
            "[N+0](a)([!#1;A,a])[A;!#1]",
            "[N+0](a)(a)a",
            "[N+0]#[A;!#1]",
            "[NH3,NH2,NH;+,+2,+3]",
            "[n+0]",
            "[n;+,+2,+3]",
            "[NH0;+,+2,+3]([A;!#1])([A;!#1])([A;!#1])[A;!#1]",
            "[NH0;+,+2,+3](=[A;!#1])([A;!#1])[!#1;A,a]",
            "[NH0;+,+2,+3](=[#6])=[#7]",
            "[N;+,+2,+3]#[A;!#1]",
            "[N;-,-2,-3]",
            "[N;+,+2,+3](=[N;-,-2,-3])=N",
            "[#7]",
            "[o]",
            "[OH,OH2]",
            "[O]([A;!#1])[A;!#1]",
            "[O](a)[!#1;A,a]",
            "[O]=[#7,#8]",
            "[OX1;-,-2,-3][#7]",
            "[OX1;-,-2,-2][#16]",
            "[O;-0]=[#16;-0]",
            "[O-]C(=O)",
            "[OX1;-,-2,-3][!#1;!N;!S]",
            "[O]=c",
            "[O]=[CH]C",
            "[O]=C(C)([A;!#1])",
            "[O]=[CH][N,O]",
            "[O]=[CH2]",
            "[O]=[CX2]=O",
            "[O]=[CH]c",
            "[O]=C([C,c])[a;!#1]",
            "[O]=C(c)[A;!#1]",
            "[O]=C([!#1;!#6])[!#1;!#6]",
            "[#8]",
            "[#9-0]",
            "[#17-0]",
            "[#35-0]",
            "[#53-0]",
            "[#9,#17,#35,#53;-]",
            "[#53;+,+2,+3]",
            "[+;#3,#11,#19,#37,#55]",
            "[#15]",
            "[S;-,-2,-3,-4,+1,+2,+3,+5,+6]",
            "[S-0]=[N,O,P,S]",
            "[S;A]",
            "[s;a]",
            "[#3,#11,#19,#37,#55]",
            "[#4,#12,#20,#38,#56]",
            "[#5,#13,#31,#49,#81]",
            "[#14,#32,#50,#82]",
            "[#33,#51,#83]",
            "[#34,#52,#84]",
            "[#21,#22,#23,#24,#25,#26,#27,#28,#29,#30]",
            "[#39,#40,#41,#42,#43,#44,#45,#46,#47,#48]",
            "[#72,#73,#74,#75,#76,#77,#78,#79,#80]",
        ]
        self._feature_names = patterns
        super().__init__(
            patterns=patterns,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        """
        Get fingerprint output feature names. They are raw SMARTS patterns
        used as feature definitions.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Ghose-Crippen feature names.
        """
        return np.asarray(self._feature_names, dtype=object)

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute Ghose-Crippen substructure fingerprints.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 307)
            Array with fingerprints.
        """
        return super().transform(X, copy)
